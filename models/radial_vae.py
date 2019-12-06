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
        self.single_reference_point = False

        # Normalizing flow layers
        self.flow_transformation = flows.Radial()

        # Amortized flow parameters
        if args.density_evaluation:
            # only performing an evaluation of flow, init flow parameters randomly
            self.alpha = nn.Parameter(torch.randn(self.num_flows, 1, 1).normal_(0, 0.1))
            self.beta = nn.Parameter(torch.randn(self.num_flows, 1, 1).normal_(0, 0.01))

            if self.single_reference_point:
                self.z_ref = nn.Parameter(torch.randn(self.z_size).fill_(0))
            else:
                self.z_ref = nn.Parameter(torch.randn(self.num_flows, self.z_size).fill_(0))

        else:
            # flow parameters learned from encoder neural network
            self.amor_alpha = nn.Linear(self.q_z_nn_output_dim, self.num_flows)
            self.amor_beta = nn.Linear(self.q_z_nn_output_dim, self.num_flows)

            if self.single_reference_point:
                self.amor_z_ref = nn.Linear(self.q_z_nn_output_dim, self.z_size)
            else:
                self.amor_z_ref = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)

        
            self.alpha, self.beta, self.z_ref = None, None, None

        self.flow_transformation = flows.Radial()

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
        if self.single_reference_point:
            self.z_ref = self.amor_z_ref(h).view(batch_size, self.z_size)
        else:
            self.z_ref = self.amor_z_ref(h).view(batch_size, self.num_flows, self.z_size)

        return mean_z, var_z

    def flow(self, z_0):
        # Initialize log-det-jacobian to zero
        log_det_jacobian = 0.0
        z = [z_0]

        for k in range(self.num_flows):
            if self.density_evaluation:
                bs = z_0.size(0)
                alpha, beta = self.alpha[k,...].expand(bs, 1, 1), self.beta[k,...].expand(bs, 1, 1)
                z_ref = self.z_ref if self.single_reference_point else self.z_ref[k,...]
                z_ref = z_ref.expand(bs, self.z_size)
            else:
                alpha, beta = self.alpha[:, k, :, :], self.beta[:, k, :, :]
                z_ref = self.z_ref if self.single_reference_point else self.z_ref[:, k, ...]
                
            z_k, ldj = self.flow_transformation(z[k], z_ref, alpha, beta)
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

