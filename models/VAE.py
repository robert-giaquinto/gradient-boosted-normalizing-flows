import numpy as np
import torch
import torch.nn as nn
import models.flows as flows
from models.layers import GatedConv2d, GatedConvTranspose2d
from utils.distributions import log_normal_standard
import random


class VAE(nn.Module):
    """
    The base VAE class containing gated convolutional encoder and decoder architecture.
    Can be used as a base class for VAE's with normalizing flows.
    """

    def __init__(self, args):
        super(VAE, self).__init__()

        # extract model settings from args
        self.z_size = args.z_size
        self.input_size = args.input_size
        self.input_type = args.input_type

        self.q_z_nn_output_dim = 256
        self.q_z_nn_hidden_dim = 128
        self.simple = True

        if self.input_size == [1, 28, 20]:
            self.last_kernel_size = (7, 5)
            self.last_pad = 2
        elif self.input_size == [3, 32, 32]:
            self.last_kernel_size = 7
            self.last_pad = 1
        else:
            self.last_kernel_size = 7
            self.last_pad = 2

        self.q_z_nn, self.q_z_mean, self.q_z_var = self.create_encoder()
        self.p_x_nn, self.p_x_mean = self.create_decoder()

        # auxiliary
        if args.cuda:
            self.FloatTensor = torch.cuda.FloatTensor
        else:
            self.FloatTensor = torch.FloatTensor

        # log-det-jacobian = 0 without flows
        self.log_det_j = self.FloatTensor(1).zero_()

    def create_encoder(self):
        """
        Helper function to create the elemental blocks for the encoder. Creates a gated convnet encoder.
        the encoder expects data as input of shape (batch_size, num_channels, width, height).
        """
        if self.input_type == 'binary':
            if self.simple:
                q_z_nn = nn.Sequential(
                    nn.Linear(np.prod(self.input_size), self.q_z_nn_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.q_z_nn_hidden_dim, self.q_z_nn_output_dim),
                    nn.Softplus(),
                )
                q_z_mean = nn.Linear(self.q_z_nn_output_dim, self.z_size)
                q_z_var = nn.Sequential(
                    nn.Linear(self.q_z_nn_output_dim, self.z_size),
                    nn.Softplus(),
                )
            else:
                q_z_nn = nn.Sequential(
                    GatedConv2d(self.input_size[0], 32, 5, 1, 2),
                    GatedConv2d(32, 32, 5, 2, 2),
                    GatedConv2d(32, 64, 5, 1, 2),
                    GatedConv2d(64, 64, 5, 2, self.last_pad),
                    GatedConv2d(64, 64, 5, 1, 2),
                    GatedConv2d(64, self.q_z_nn_output_dim, self.last_kernel_size, 1, 0),
                )
                q_z_mean = nn.Linear(self.q_z_nn_output_dim, self.z_size)
                q_z_var = nn.Sequential(
                    nn.Linear(self.q_z_nn_output_dim, self.z_size),
                    nn.Softplus(),
                )
            return q_z_nn, q_z_mean, q_z_var

        elif self.input_type == 'multinomial':
            act = None
            q_z_nn = nn.Sequential(
                GatedConv2d(self.input_size[0], 32, 5, 1, 2, activation=act),
                GatedConv2d(32, 32, 5, 2, 2, activation=act),
                GatedConv2d(32, 64, 5, 1, 2, activation=act),
                GatedConv2d(64, 64, 5, 2, self.last_pad, activation=act),
                GatedConv2d(64, 64, 5, 1, 2, activation=act),
                GatedConv2d(64, 256, self.last_kernel_size, 1, 0, activation=act)
            )
            q_z_mean = nn.Linear(256, self.z_size)
            q_z_var = nn.Sequential(
                nn.Linear(256, self.z_size),
                nn.Softplus(),
                nn.Hardtanh(min_val=0.01, max_val=7.)

            )
            return q_z_nn, q_z_mean, q_z_var

    def create_decoder(self):
        """
        Helper function to create the elemental blocks for the decoder. Creates a gated convnet decoder.
        """

        num_classes = 256

        if self.input_type == 'binary':
            if self.simple:
                p_x_nn = nn.Sequential(
                    nn.Linear(self.z_size, self.q_z_nn_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.q_z_nn_hidden_dim, self.q_z_nn_output_dim),
                    nn.Softplus(),
                )
                p_x_mean = nn.Sequential(
                    nn.Linear(self.q_z_nn_output_dim, np.prod(self.input_size)),
                    nn.Sigmoid()
                )
            else:
                p_x_nn = nn.Sequential(
                    GatedConvTranspose2d(self.z_size, 64, self.last_kernel_size, 1, 0),
                    GatedConvTranspose2d(64, 64, 5, 1, 2),
                    GatedConvTranspose2d(64, 32, 5, 2, 2, 1),
                    GatedConvTranspose2d(32, 32, 5, 1, 2),
                    GatedConvTranspose2d(32, 32, 5, 2, 2, 1),
                    GatedConvTranspose2d(32, 32, 5, 1, 2)
                )
                p_x_mean = nn.Sequential(
                    nn.Conv2d(32, self.input_size[0], 1, 1, 0),
                    nn.Sigmoid()
                )
            return p_x_nn, p_x_mean

        elif self.input_type == 'multinomial':
            act = None
            p_x_nn = nn.Sequential(
                GatedConvTranspose2d(self.z_size, 64, self.last_kernel_size, 1, 0, activation=act),
                GatedConvTranspose2d(64, 64, 5, 1, 2, activation=act),
                GatedConvTranspose2d(64, 32, 5, 2, self.last_pad, 1, activation=act),
                GatedConvTranspose2d(32, 32, 5, 1, 2, activation=act),
                GatedConvTranspose2d(32, 32, 5, 2, 2, 1, activation=act),
                GatedConvTranspose2d(32, 32, 5, 1, 2, activation=act)
            )
            p_x_mean = nn.Sequential(
                nn.Conv2d(32, 256, 5, 1, 2),
                nn.Conv2d(256, self.input_size[0] * num_classes, 1, 1, 0),
                # output shape: batch_size, num_channels * num_classes, pixel_width, pixel_height
            )

            return p_x_nn, p_x_mean

        else:
            raise ValueError('invalid input type!!')

    def reparameterize(self, mu, var):
        """
        Samples z from a multivariate Gaussian with diagonal covariance matrix using the
         reparameterization trick.
        """
        std = var.sqrt() # multiply by 2?
        eps = self.FloatTensor(std.size()).normal_()
        z = eps.mul(std).add_(mu)
        return z

    def encode(self, x):
        """
        Encoder expects following data shapes as input: shape = (batch_size, num_channels, width, height)
        """
        h = self.q_z_nn(x)
        if not self.simple:
            h = h.view(h.size(0), -1)
        mean = self.q_z_mean(h)
        var = self.q_z_var(h)
        return mean, var

    def decode(self, z):
        """
        Decoder outputs reconstructed image in the following shapes:
        x_mean.shape = (batch_size, num_channels, width, height)
        """
        if not self.simple:
            z = z.view(z.size(0), self.z_size, 1, 1)
        h = self.p_x_nn(z)
        x_mean = self.p_x_mean(h)
        return x_mean

    def forward(self, x):
        """
        Evaluates the model as a whole, encodes and decodes. Note that the log det jacobian is zero
         for a plain VAE (without flows), and z_0 = z_k.
        """
        # mean and variance of z
        z_mu, z_var = self.encode(x)
        # sample z
        z = self.reparameterize(z_mu, z_var)
        x_mean = self.decode(z)
        return x_mean, z_mu, z_var, self.log_det_j, z, z



class BaggedVAE(VAE):
    """
    Variational auto-encoder with bagged planar flows in the encoder.

    """

    def __init__(self, args):
        super(BaggedVAE, self).__init__(args)

        # boosting parameters
        self.num_learners = args.num_learners
        self.last_learner_trained = None

        # Initialize log-det-jacobian to zero
        self.log_det_j = self.FloatTensor(1, 1).fill_(0.0)

        # Flow parameters
        if args.learner_type == "planar":
            flow = flows.Planar
        else:
            raise ValueError("Lets keep it simple for now and only implement planar weak learners")
        self.num_flows = args.num_flows

        # Amortized flow parameters for each weak learner
        for c in range(self.num_learners):
            amor_u_c = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)
            amor_w_c = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)
            amor_b_c = nn.Linear(self.q_z_nn_output_dim, self.num_flows)
            self.add_module('amor_u_' + str(c), amor_u_c)
            self.add_module('amor_w_' + str(c), amor_w_c)
            self.add_module('amor_b_' + str(c), amor_b_c)

        # Normalizing flow layers
        for c in range(self.num_learners):
            for k in range(self.num_flows):
                flow_c_k = flow()
                self.add_module('flow_' + str(c) + '_' + str(k), flow_c_k)

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        batch_size = x.size(0)

        h = self.q_z_nn(x)
        h = h.view(-1, self.q_z_nn_output_dim)
        mean_z = self.q_z_mean(h)
        var_z = self.q_z_var(h)

        # return amortized u an w for all flows
        u, w, b = [], [], []
        for c in range(self.num_learners):
            u_c = getattr(self, 'amor_u_' + str(c))
            w_c = getattr(self, 'amor_w_' + str(c))
            b_c = getattr(self, 'amor_b_' + str(c))
            u.append(u_c(h).view(batch_size, self.num_flows, self.z_size, 1))
            w.append(w_c(h).view(batch_size, self.num_flows, 1, self.z_size))
            b.append(b_c(h).view(batch_size, self.num_flows, 1, 1))

        return mean_z, var_z, u, w, b

    def forward(self, x):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0 [sum_k log |det dz_k / dz_k-1| ].
        """
        z_mu, z_var, u, w, b = self.encode(x)
        z_0 = self.reparameterize(z_mu, z_var)
        self.log_det_j = self.FloatTensor(x.size(0)).fill_(0.0)

        if self.training:
            # Normalizing flows for training: train one weak learner at a time
            c = random.randint(0, self.num_learners - 1)
            Z_arr = [z_0]
            # apply flow transformations
            for k in range(self.num_flows):
                flow_c_k = getattr(self, 'flow_' + str(c) + '_' + str(k))
                z_k, ldj = flow_c_k(Z_arr[k], u[c][:, k, :, :], w[c][:, k, :, :], b[c][:, k, :, :])
                Z_arr.append(z_k)
                self.log_det_j += ldj

            z_out = Z_arr[-1]
            self.last_learner_trained = c

        else:
            # Normalizing flows for prediction: aggregate over all learners
            z_sum = torch.zeros_like(z_0)  # defaults to device of z_0

            for c in range(self.num_learners):
                # Sample a z_0 for this learner or sample one z_0 for all learners?
                Z_c = [z_0]

                # apply flow transformations
                for k in range(self.num_flows):
                    flow_c_k = getattr(self, 'flow_' + str(c) + '_' + str(k))
                    z_ck, ldj = flow_c_k(Z_c[k], u[c][:, k, :, :], w[c][:, k, :, :], b[c][:, k, :, :])
                    Z_c.append(z_ck)
                    self.log_det_j += ldj

                # accumulate final transformation from each weak learner
                # TODO: should these be accumulate with a weighting scheme?
                z_sum = z_sum + Z_c[-1]

            # aggregate estimates of z_k across each weak learner
            z_out = (1.0 / self.num_learners) * z_sum

        # decode aggregated output of weak learners
        x_mean = self.decode(z_out)

        return x_mean, z_mu, z_var, self.log_det_j, z_0, z_out




class BoostedVAE(VAE):
    """
    Variational auto-encoder with boosted planar flows in the encoder.

    """

    def __init__(self, args):
        super(BoostedVAE, self).__init__(args)

        # boosting parameters
        self.num_learners = args.num_learners
        self.last_learner_trained = None

        # Initialize log-det-jacobian to zero
        self.log_det_j = self.FloatTensor(1, 1).fill_(0.0)

        # Flow parameters
        if args.learner_type == "planar":
            flow = flows.Planar
        else:
            raise ValueError("Lets keep it simple for now and only implement planar weak learners")
        self.num_flows = args.num_flows

        # Amortized flow parameters for each weak learner
        for c in range(self.num_learners):
            amor_u_c = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)
            amor_w_c = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)
            amor_b_c = nn.Linear(self.q_z_nn_output_dim, self.num_flows)
            self.add_module('amor_u_' + str(c), amor_u_c)
            self.add_module('amor_w_' + str(c), amor_w_c)
            self.add_module('amor_b_' + str(c), amor_b_c)

        # Normalizing flow layers
        for c in range(self.num_learners):
            for k in range(self.num_flows):
                flow_c_k = flow()
                self.add_module('flow_' + str(c) + '_' + str(k), flow_c_k)

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        batch_size = x.size(0)

        h = self.q_z_nn(x)
        h = h.view(-1, self.q_z_nn_output_dim)
        mean_z = self.q_z_mean(h)
        var_z = self.q_z_var(h)

        # return amortized u an w for all flows
        u, w, b = [], [], []
        for c in range(self.num_learners):
            u_c = getattr(self, 'amor_u_' + str(c))
            w_c = getattr(self, 'amor_w_' + str(c))
            b_c = getattr(self, 'amor_b_' + str(c))
            u.append(u_c(h).view(batch_size, self.num_flows, self.z_size, 1))
            w.append(w_c(h).view(batch_size, self.num_flows, 1, self.z_size))
            b.append(b_c(h).view(batch_size, self.num_flows, 1, 1))

        return mean_z, var_z, u, w, b

    def forward(self, x):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0 [sum_k log |det dz_k / dz_k-1| ].
        """
        self.log_det_j = self.FloatTensor(self.num_learners, x.size(0)).fill_(0.0)
        z_mu, z_var, u, w, b = self.encode(x)
        z_0 = self.reparameterize(z_mu, z_var)

        # Normalizing flows
        z_sum = torch.zeros_like(z_0)  # defaults to device of z_0
        for c in range(self.num_learners):
            # Sample a z_0 for this learner or sample one z_0 for all learners?
            Z_c = [z_0]

            # apply flow transformations
            for k in range(self.num_flows):
                flow_c_k = getattr(self, 'flow_' + str(c) + '_' + str(k))
                z_ck, ldj = flow_c_k(Z_c[k], u[c][:, k, :, :], w[c][:, k, :, :], b[c][:, k, :, :])
                Z_c.append(z_ck)
                self.log_det_j[c, :] += ldj

            # accumulate final transformation from each weak learner
            # TODO: should these be accumulate with a weighting scheme?
            z_sum = z_sum + Z_c[-1]

        # aggregate estimates of z_k across each weak learner
        z = (1.0 / self.num_learners) * z_sum

        # decode aggregated output of weak learners
        x_mean = self.decode(z)

        if self.training:
            # only concerned about log det jacobian of learner being trained
            self.last_learner_trained = random.randint(0, self.num_learners - 1)
            self.log_det_j = self.log_det_j[self.last_learner_trained, :]
        else:
            self.log_det_j = torch.sum(self.log_det_j, dim=0)

        return x_mean, z_mu, z_var, self.log_det_j, z_0, z




class PlanarVAE(VAE):
    """
    Variational auto-encoder with planar flows in the encoder.
    """

    def __init__(self, args):
        super(PlanarVAE, self).__init__(args)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        # Flow parameters
        flow = flows.Planar
        self.num_flows = args.num_flows

        # Amortized flow parameters
        self.amor_u = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)
        self.amor_w = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)
        self.amor_b = nn.Linear(self.q_z_nn_output_dim, self.num_flows)

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow()
            self.add_module('flow_' + str(k), flow_k)

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        batch_size = x.size(0)

        h = self.q_z_nn(x)
        h = h.view(-1, self.q_z_nn_output_dim)
        mean_z = self.q_z_mean(h)
        var_z = self.q_z_var(h)

        # return amortized u an w for all flows
        u = self.amor_u(h).view(batch_size, self.num_flows, self.z_size, 1)
        w = self.amor_w(h).view(batch_size, self.num_flows, 1, self.z_size)
        b = self.amor_b(h).view(batch_size, self.num_flows, 1, 1)

        return mean_z, var_z, u, w, b

    def forward(self, x):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        self.log_det_j = 0.

        z_mu, z_var, u, w, b = self.encode(x)

        # Sample z_0
        z = [self.reparameterize(z_mu, z_var)]

        # Normalizing flows
        for k in range(self.num_flows):
            flow_k = getattr(self, 'flow_' + str(k))
            z_k, log_det_jacobian = flow_k(z[k], u[:, k, :, :], w[:, k, :, :], b[:, k, :, :])
            z.append(z_k)
            self.log_det_j += log_det_jacobian

        x_mean = self.decode(z[-1])

        return x_mean, z_mu, z_var, self.log_det_j, z[0], z[-1]


class RadialVAE(VAE):
    """
    Variational auto-encoder with planar flows in the encoder.
    """

    def __init__(self, args):
        super(RadialVAE, self).__init__(args)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        # Flow parameters
        flow = flows.Radial
        self.num_flows = args.num_flows

        # Amortized flow parameters
        #self.amor_alpha = nn.Linear(self.q_z_nn_output_dim, self.num_flows)
        self.amor_alpha = nn.Sequential(
                nn.Linear(self.q_z_nn_output_dim, self.num_flows),
                nn.Softplus(),
                nn.Hardtanh(min_val=0.01, max_val=7.)
        )
        self.amor_beta = nn.Linear(self.q_z_nn_output_dim, self.num_flows)

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow(z_size=self.z_size)
            self.add_module('flow_' + str(k), flow_k)

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """
        batch_size = x.size(0)

        h = self.q_z_nn(x)
        h = h.view(-1, self.q_z_nn_output_dim)
        mean_z = self.q_z_mean(h)
        var_z = self.q_z_var(h)

        # return amortized u an w for all flows
        alpha = self.amor_alpha(h).view(batch_size, self.num_flows, 1, 1)
        #alpha = nn.functional.relu(self.amor_alpha(h).view(batch_size, self.num_flows, 1, 1))
        #alpha = nn.functional.hardtanh(self.amor_alpha(h).view(batch_size, self.num_flows, 1, 1), min_val=0.0, max_val=2.0)
        beta = self.amor_beta(h).view(batch_size, self.num_flows, 1, 1)
        return mean_z, var_z, alpha, beta

    def forward(self, x):
        """
        Forward pass with radial flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """
        self.log_det_j = 0.

        z_mu, z_var, alpha, beta = self.encode(x)

        # Sample z_0
        z = [self.reparameterize(z_mu, z_var)]

        # create reference point z0
        z0 = self.reparameterize(z_mu, z_var).mean(dim=0).unsqueeze(0)

        # Normalizing flows
        for k in range(self.num_flows):
            flow_k = getattr(self, 'flow_' + str(k))
            z_k, log_det_jacobian = flow_k(z[k], z0, alpha[:, k, :, :], beta[:, k, :, :])
            z.append(z_k)
            self.log_det_j += log_det_jacobian

        x_mean = self.decode(z[-1])

        return x_mean, z_mu, z_var, self.log_det_j, z[0], z[-1]


class OrthogonalSylvesterVAE(VAE):
    """
    Variational auto-encoder with orthogonal flows in the encoder.
    """

    def __init__(self, args):
        super(OrthogonalSylvesterVAE, self).__init__(args)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        # Flow parameters
        flow = flows.Sylvester
        self.num_flows = args.num_flows
        self.num_ortho_vecs = args.num_ortho_vecs

        assert (self.num_ortho_vecs <= self.z_size) and (self.num_ortho_vecs > 0)

        # Orthogonalization parameters
        if self.num_ortho_vecs == self.z_size:
            self.cond = 1.e-5
        else:
            self.cond = 1.e-6

        self.steps = 100
        identity = torch.eye(self.num_ortho_vecs, self.num_ortho_vecs)
        # Add batch dimension
        identity = identity.unsqueeze(0)
        # Put identity in buffer so that it will be moved to GPU if needed by any call of .cuda
        self.register_buffer('_eye', identity)  #self.register_buffer('_eye', Variable(identity))
        self._eye.requires_grad = False

        # Masks needed for triangular R1 and R2.
        triu_mask = torch.triu(torch.ones(self.num_ortho_vecs, self.num_ortho_vecs), diagonal=1)
        triu_mask = triu_mask.unsqueeze(0).unsqueeze(3)
        diag_idx = torch.arange(0, self.num_ortho_vecs).long()

        self.register_buffer('triu_mask', triu_mask)  #self.register_buffer('triu_mask', Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer('diag_idx', diag_idx)

        # Amortized flow parameters
        # Diagonal elements of R1 * R2 have to satisfy -1 < R1 * R2 for flow to be invertible
        self.diag_activation = nn.Tanh()

        self.amor_d = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.num_ortho_vecs * self.num_ortho_vecs)

        self.amor_diag1 = nn.Sequential(
            nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.num_ortho_vecs),
            self.diag_activation
        )
        self.amor_diag2 = nn.Sequential(
            nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.num_ortho_vecs),
            self.diag_activation
        )

        self.amor_q = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size * self.num_ortho_vecs)
        self.amor_b = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.num_ortho_vecs)

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow(self.num_ortho_vecs)
            self.add_module('flow_' + str(k), flow_k)

    def batch_construct_orthogonal(self, q):
        """
        Batch orthogonal matrix construction.
        :param q:  q contains batches of matrices, shape : (batch_size * num_flows, z_size * num_ortho_vecs)
        :return: batches of orthogonalized matrices, shape: (batch_size * num_flows, z_size, num_ortho_vecs)
        """

        # Reshape to shape (num_flows * batch_size, z_size * num_ortho_vecs)
        q = q.view(-1, self.z_size * self.num_ortho_vecs)

        norm = torch.norm(q, p=2, dim=1, keepdim=True)
        amat = torch.div(q, norm)
        dim0 = amat.size(0)
        amat = amat.resize(dim0, self.z_size, self.num_ortho_vecs)

        max_norm = 0.

        # Iterative orthogonalization
        for s in range(self.steps):
            tmp = torch.bmm(amat.transpose(2, 1), amat)
            tmp = self._eye - tmp
            tmp = self._eye + 0.5 * tmp
            amat = torch.bmm(amat, tmp)

            # Testing for convergence
            test = torch.bmm(amat.transpose(2, 1), amat) - self._eye
            norms2 = torch.sum(torch.norm(test, p=2, dim=2) ** 2, dim=1)
            norms = torch.sqrt(norms2)
            max_norm = torch.max(norms).data[0]
            if max_norm <= self.cond:
                break

        if max_norm > self.cond:
            print('\nWARNING WARNING WARNING: orthogonalization not complete')
            print('\t Final max norm =', max_norm)

            print()

        # Reshaping: first dimension is batch_size
        amat = amat.view(-1, self.num_flows, self.z_size, self.num_ortho_vecs)
        amat = amat.transpose(0, 1)

        return amat

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        batch_size = x.size(0)

        h = self.q_z_nn(x)
        h = h.view(-1, self.q_z_nn_output_dim)
        mean_z = self.q_z_mean(h)
        var_z = self.q_z_var(h)

        # Amortized r1, r2, q, b for all flows

        full_d = self.amor_d(h)
        diag1 = self.amor_diag1(h)
        diag2 = self.amor_diag2(h)

        full_d = full_d.resize(batch_size, self.num_ortho_vecs, self.num_ortho_vecs, self.num_flows)
        diag1 = diag1.resize(batch_size, self.num_ortho_vecs, self.num_flows)
        diag2 = diag2.resize(batch_size, self.num_ortho_vecs, self.num_flows)

        r1 = full_d * self.triu_mask
        r2 = full_d.transpose(2, 1) * self.triu_mask

        r1[:, self.diag_idx, self.diag_idx, :] = diag1
        r2[:, self.diag_idx, self.diag_idx, :] = diag2

        q = self.amor_q(h)
        b = self.amor_b(h)

        # Resize flow parameters to divide over K flows
        b = b.resize(batch_size, 1, self.num_ortho_vecs, self.num_flows)

        return mean_z, var_z, r1, r2, q, b

    def forward(self, x):
        """
        Forward pass with orthogonal sylvester flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        self.log_det_j = 0.

        z_mu, z_var, r1, r2, q, b = self.encode(x)

        # Orthogonalize all q matrices
        q_ortho = self.batch_construct_orthogonal(q)

        # Sample z_0
        z = [self.reparameterize(z_mu, z_var)]

        # Normalizing flows
        for k in range(self.num_flows):

            flow_k = getattr(self, 'flow_' + str(k))
            z_k, log_det_jacobian = flow_k(z[k], r1[:, :, :, k], r2[:, :, :, k], q_ortho[k, :, :, :], b[:, :, :, k])

            z.append(z_k)
            self.log_det_j += log_det_jacobian

        x_mean = self.decode(z[-1])

        return x_mean, z_mu, z_var, self.log_det_j, z[0], z[-1]


class HouseholderSylvesterVAE(VAE):
    """
    Variational auto-encoder with householder sylvester flows in the encoder.
    """

    def __init__(self, args):
        super(HouseholderSylvesterVAE, self).__init__(args)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        # Flow parameters
        flow = flows.Sylvester
        self.num_flows = args.num_flows
        self.num_householder = args.num_householder
        assert self.num_householder > 0

        identity = torch.eye(self.z_size, self.z_size)
        # Add batch dimension
        identity = identity.unsqueeze(0)
        # Put identity in buffer so that it will be moved to GPU if needed by any call of .cuda
        self.register_buffer('_eye', identity)  #self.register_buffer('_eye', Variable(identity))
        self._eye.requires_grad = False

        # Masks needed for triangular r1 and r2.
        triu_mask = torch.triu(torch.ones(self.z_size, self.z_size), diagonal=1)
        triu_mask = triu_mask.unsqueeze(0).unsqueeze(3)
        diag_idx = torch.arange(0, self.z_size).long()

        self.register_buffer('triu_mask', triu_mask)  #self.register_buffer('triu_mask', Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer('diag_idx', diag_idx)

        # Amortized flow parameters
        # Diagonal elements of r1 * r2 have to satisfy -1 < r1 * r2 for flow to be invertible
        self.diag_activation = nn.Tanh()

        self.amor_d = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size * self.z_size)

        self.amor_diag1 = nn.Sequential(
            nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size),
            self.diag_activation
        )
        self.amor_diag2 = nn.Sequential(
            nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size),
            self.diag_activation
        )

        self.amor_q = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size * self.num_householder)

        self.amor_b = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow(self.z_size)

            self.add_module('flow_' + str(k), flow_k)

    def batch_construct_orthogonal(self, q):
        """
        Batch orthogonal matrix construction.
        :param q:  q contains batches of matrices, shape : (batch_size, num_flows * z_size * num_householder)
        :return: batches of orthogonalized matrices, shape: (batch_size * num_flows, z_size, z_size)
        """

        # Reshape to shape (num_flows * batch_size * num_householder, z_size)
        q = q.view(-1, self.z_size)

        norm = torch.norm(q, p=2, dim=1, keepdim=True)   # ||v||_2
        v = torch.div(q, norm)  # v / ||v||_2

        # Calculate Householder Matrices
        vvT = torch.bmm(v.unsqueeze(2), v.unsqueeze(1))  # v * v_T : batch_dot( B x L x 1 * B x 1 x L ) = B x L x L

        amat = self._eye - 2 * vvT  # NOTICE: v is already normalized! so there is no need to calculate vvT/vTv

        # Reshaping: first dimension is batch_size * num_flows
        amat = amat.view(-1, self.num_householder, self.z_size, self.z_size)

        tmp = amat[:, 0]
        for k in range(1, self.num_householder):
            tmp = torch.bmm(amat[:, k], tmp)

        amat = tmp.view(-1, self.num_flows, self.z_size, self.z_size)
        amat = amat.transpose(0, 1)

        return amat

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        batch_size = x.size(0)

        h = self.q_z_nn(x)
        h = h.view(-1, self.q_z_nn_output_dim)
        mean_z = self.q_z_mean(h)
        var_z = self.q_z_var(h)

        # Amortized r1, r2, q, b for all flows
        full_d = self.amor_d(h)
        diag1 = self.amor_diag1(h)
        diag2 = self.amor_diag2(h)

        full_d = full_d.resize(batch_size, self.z_size, self.z_size, self.num_flows)
        diag1 = diag1.resize(batch_size, self.z_size, self.num_flows)
        diag2 = diag2.resize(batch_size, self.z_size, self.num_flows)

        r1 = full_d * self.triu_mask
        r2 = full_d.transpose(2, 1) * self.triu_mask

        r1[:, self.diag_idx, self.diag_idx, :] = diag1
        r2[:, self.diag_idx, self.diag_idx, :] = diag2

        q = self.amor_q(h)

        b = self.amor_b(h)

        # Resize flow parameters to divide over K flows
        b = b.resize(batch_size, 1, self.z_size, self.num_flows)

        return mean_z, var_z, r1, r2, q, b

    def forward(self, x):
        """
        Forward pass with orthogonal flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        self.log_det_j = 0.
        batch_size = x.size(0)

        z_mu, z_var, r1, r2, q, b = self.encode(x)

        # Orthogonalize all q matrices
        q_ortho = self.batch_construct_orthogonal(q)

        # Sample z_0
        z = [self.reparameterize(z_mu, z_var)]

        # Normalizing flows
        for k in range(self.num_flows):

            flow_k = getattr(self, 'flow_' + str(k))
            q_k = q_ortho[k]

            z_k, log_det_jacobian = flow_k(z[k], r1[:, :, :, k], r2[:, :, :, k], q_k, b[:, :, :, k], sum_ldj=True)

            z.append(z_k)
            self.log_det_j += log_det_jacobian

        x_mean = self.decode(z[-1])

        return x_mean, z_mu, z_var, self.log_det_j, z[0], z[-1]


class TriangularSylvesterVAE(VAE):
    """
    Variational auto-encoder with triangular Sylvester flows in the encoder. Alternates between setting
    the orthogonal matrix equal to permutation and identity matrix for each flow.
    """

    def __init__(self, args):
        super(TriangularSylvesterVAE, self).__init__(args)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        # Flow parameters
        flow = flows.TriangularSylvester
        self.num_flows = args.num_flows

        # permuting indices corresponding to Q=P (permutation matrix) for every other flow
        flip_idx = torch.arange(self.z_size - 1, -1, -1).long()
        self.register_buffer('flip_idx', flip_idx)

        # Masks needed for triangular r1 and r2.
        triu_mask = torch.triu(torch.ones(self.z_size, self.z_size), diagonal=1)
        triu_mask = triu_mask.unsqueeze(0).unsqueeze(3)
        diag_idx = torch.arange(0, self.z_size).long()

        self.register_buffer('triu_mask', triu_mask)  #self.register_buffer('triu_mask', Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer('diag_idx', diag_idx)

        # Amortized flow parameters
        # Diagonal elements of r1 * r2 have to satisfy -1 < r1 * r2 for flow to be invertible
        self.diag_activation = nn.Tanh()

        self.amor_d = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size * self.z_size)

        self.amor_diag1 = nn.Sequential(
            nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size),
            self.diag_activation
        )
        self.amor_diag2 = nn.Sequential(
            nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size),
            self.diag_activation
        )

        self.amor_b = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow(self.z_size)

            self.add_module('flow_' + str(k), flow_k)

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        batch_size = x.size(0)

        h = self.q_z_nn(x)
        h = h.view(-1, self.q_z_nn_output_dim)
        mean_z = self.q_z_mean(h)
        var_z = self.q_z_var(h)

        # Amortized r1, r2, b for all flows
        full_d = self.amor_d(h)
        diag1 = self.amor_diag1(h)
        diag2 = self.amor_diag2(h)

        full_d = full_d.resize(batch_size, self.z_size, self.z_size, self.num_flows)
        diag1 = diag1.resize(batch_size, self.z_size, self.num_flows)
        diag2 = diag2.resize(batch_size, self.z_size, self.num_flows)

        r1 = full_d * self.triu_mask
        r2 = full_d.transpose(2, 1) * self.triu_mask

        r1[:, self.diag_idx, self.diag_idx, :] = diag1
        r2[:, self.diag_idx, self.diag_idx, :] = diag2

        b = self.amor_b(h)

        # Resize flow parameters to divide over K flows
        b = b.resize(batch_size, 1, self.z_size, self.num_flows)

        return mean_z, var_z, r1, r2, b

    def forward(self, x):
        """
        Forward pass with orthogonal flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        self.log_det_j = 0.

        z_mu, z_var, r1, r2, b = self.encode(x)

        # Sample z_0
        z = [self.reparameterize(z_mu, z_var)]

        # Normalizing flows
        for k in range(self.num_flows):

            flow_k = getattr(self, 'flow_' + str(k))
            if k % 2 == 1:
                # Alternate with reorderering z for triangular flow
                permute_z = self.flip_idx
            else:
                permute_z = None

            z_k, log_det_jacobian = flow_k(z[k], r1[:, :, :, k], r2[:, :, :, k], b[:, :, :, k], permute_z, sum_ldj=True)

            z.append(z_k)
            self.log_det_j += log_det_jacobian

        x_mean = self.decode(z[-1])

        return x_mean, z_mu, z_var, self.log_det_j, z[0], z[-1]


class IAFVAE(VAE):
    """
    Variational auto-encoder with inverse autoregressive flows in the encoder.
    """

    def __init__(self, args):
        super(IAFVAE, self).__init__(args)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.
        self.h_size = args.made_h_size

        self.h_context = nn.Linear(self.q_z_nn_output_dim, self.h_size)

        # Flow parameters
        self.num_flows = args.num_flows
        self.flow = flows.IAF(z_size=self.z_size, num_flows=self.num_flows,
                              num_hidden=1, h_size=self.h_size, conv2d=False)

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and context h for flows.
        """

        h = self.q_z_nn(x)
        h = h.view(-1, self.q_z_nn_output_dim)
        mean_z = self.q_z_mean(h)
        var_z = self.q_z_var(h)
        h_context = self.h_context(h)

        return mean_z, var_z, h_context

    def forward(self, x):
        """
        Forward pass with inverse autoregressive flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        # mean and variance of z
        z_mu, z_var, h_context = self.encode(x)
        # sample z
        z_0 = self.reparameterize(z_mu, z_var)

        # iaf flows
        z_k, self.log_det_j = self.flow(z_0, h_context)

        # decode
        x_mean = self.decode(z_k)

        return x_mean, z_mu, z_var, self.log_det_j, z_0, z_k
