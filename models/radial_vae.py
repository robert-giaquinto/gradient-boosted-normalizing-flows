import numpy as np
import torch
import torch.nn as nn
import random

from models.vae import VAE
import models.flows as flows


class RadialVAE(VAE):
    """
    Variational auto-encoder with planar flows in the encoder.
    """

    def __init__(self, args):
        super(RadialVAE, self).__init__(args)
        self.num_flows = args.num_flows
        self.density_evaluation = args.density_evaluation

        # Normalizing flow layers
        self.flow_transformation = flows.Radial()

        # Amortized flow parameters
        if args.density_evaluation:
            self.q_z_nn, self.q_z_mean, self.q_z_var = None, None, None
            self.p_x_nn, self.p_x_mean = None, None

            # only performing an evaluation of flow, init flow parameters randomly
            self.alpha = nn.Parameter(torch.randn(args.batch_size, self.num_flows, 1, 1).normal_(0, 0.01))
            self.beta = nn.Parameter(torch.randn(args.batch_size, self.num_flows, 1, 1).normal_(0, 0.01))
            self.z_ref = nn.Parameter(torch.randn(args.batch_size, self.z_size).fill_(0))

            self.z_mu = nn.Parameter(torch.zeros(2)).requires_grad_(True)
            self.z_var = nn.Parameter(torch.ones(2)).requires_grad_(True)

        else:
            # flow parameters learned from encoder neural network
            self.amor_alpha = nn.Sequential(
                nn.Linear(self.q_z_nn_output_dim, self.num_flows),
                nn.Softplus(),
                nn.Hardtanh(min_val=0.01, max_val=7.)
            )
            self.amor_beta = nn.Linear(self.q_z_nn_output_dim, self.num_flows)
            self.amor_z_ref = nn.Linear(self.q_z_nn_output_dim, self.z_size)
        
            self.alpha, self.beta, self.z_ref = None, None, None

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
        self.alpha = self.amor_alpha(h).view(batch_size, self.num_flows, 1, 1)
        self.beta = self.amor_beta(h).view(batch_size, self.num_flows, 1, 1)
        self.z_ref = self.amor_z_ref(h).view(batch_size, self.z_size)
        return mean_z, var_z

    def flow(self, z_0):
        # Initialize log-det-jacobian to zero
        log_det_jacobian = 0.0

        if self.density_evaluation:
            # use trainable base distribution parameters
            # treat z_0 as the "noise"
            z = [self.z_mu + self.z_var * z_0]
        else:
            # already given the reparameterized z_0
            z = [z_0]

        for k in range(self.num_flows):
            z_k, ldj = self.flow_transformation(z[k], self.z_ref, self.alpha[:, k, :, :], self.beta[:, k, :, :])
            z.append(z_k)
            log_det_jacobian += ldj

        return z[-1], log_det_jacobian

    def forward(self, x):
        """
        Forward pass with radial flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """
        z_mu, z_var = self.encode(x)

        # Sample z_0
        z_0 = self.reparameterize(z_mu, z_var)

        # pass through normalizing flow
        z_k, log_det_jacobian = self.flow(z_0)

        # reconstruct
        x_recon = self.decode(z_k)

        return x_recon, z_mu, z_var, log_det_jacobian, z_0, z_k

