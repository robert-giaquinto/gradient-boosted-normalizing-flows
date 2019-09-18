import numpy as np
import torch
import torch.nn as nn
import random

from models.vae import VAE
import models.flows as flows


class PlanarVAE(VAE):
    """
    Variational auto-encoder with planar flows in the encoder.
    """

    def __init__(self, args):
        super(PlanarVAE, self).__init__(args)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        # Flow parameters
        self.num_flows = args.num_flows

        # Amortized flow parameters
        self.amor_u = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)
        self.amor_w = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)
        self.amor_b = nn.Linear(self.q_z_nn_output_dim, self.num_flows)

        # Normalizing flow layers
        #flow = flows.Planar
        #for k in range(self.num_flows):
        #    flow_k = flow()
        #    self.add_module('flow_' + str(k), flow_k)
        self.flow = flows.Planar()

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
            #flow_k = getattr(self, 'flow_' + str(k))
            #z_k, log_det_jacobian = flow_k(z[k], u[:, k, :, :], w[:, k, :, :], b[:, k, :, :])
            z_k, log_det_jacobian = self.flow(z[k], u[:, k, :, :], w[:, k, :, :], b[:, k, :, :])
            z.append(z_k)
            self.log_det_j += log_det_jacobian

        x_mean = self.decode(z[-1])

        return x_mean, z_mu, z_var, self.log_det_j, z[0], z[-1]

