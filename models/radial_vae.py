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

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        # Flow parameters
        flow = flows.Radial
        self.num_flows = args.num_flows

        # Amortized flow parameters
        self.amor_alpha = nn.Sequential(
            nn.Linear(self.q_z_nn_output_dim, self.num_flows),
            nn.Softplus(),
            nn.Hardtanh(min_val=0.01, max_val=7.)
        )
        self.amor_beta = nn.Linear(self.q_z_nn_output_dim, self.num_flows)
        self.amor_z_ref = nn.Linear(self.q_z_nn_output_dim, self.z_size)

        self.flow = flows.Radial()

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
        beta = self.amor_beta(h).view(batch_size, self.num_flows, 1, 1)
        z_ref = self.amor_z_ref(h).view(batch_size, self.z_size)
        return mean_z, var_z, alpha, beta, z_ref

    def forward(self, x):
        """
        Forward pass with radial flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """
        self.log_det_j = 0.

        z_mu, z_var, alpha, beta, z_ref = self.encode(x)

        # Sample z_0
        z = [self.reparameterize(z_mu, z_var)]

        # create reference point z0
        # better to create a running mean of z0 as reference point?
        # z0 = self.reparameterize(z_mu, z_var).mean(dim=0).unsqueeze(0)

        # Normalizing flows
        for k in range(self.num_flows):
            z_k, log_det_jacobian = self.flow(z[k], z_ref, alpha[:, k, :, :], beta[:, k, :, :])
            z.append(z_k)
            self.log_det_j += log_det_jacobian

        x_mean = self.decode(z[-1])

        return x_mean, z_mu, z_var, self.log_det_j, z[0], z[-1]

