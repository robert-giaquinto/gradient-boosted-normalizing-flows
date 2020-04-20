import numpy as np
import torch
import torch.nn as nn
import random
import torch.distributions as D

from models.vae import VAE
from models.generative_flow import GenerativeFlow
import models.transformations as flows


class PlanarFlow(GenerativeFlow):
    """
    Generative flow using planar transformations
    """

    def __init__(self, args):
        super(PlanarFlow, self).__init__()

        self.u = nn.Parameter(torch.randn(self.num_flows, self.z_size, 1).normal_(0, 0.01))
        self.w = nn.Parameter(torch.randn(self.num_flows, 1, self.z_size).normal_(0, 0.01))
        self.b = nn.Parameter(torch.randn(self.num_flows, 1, 1).fill_(0))
        self.flow_transformation = flows.Planar()

    def flow(self, z_0):
        return self.forward(z_0)

    def forward(self, z_0):
        log_det_jacobian = 0.0
        z = [z_0]
        for k in range(self.num_flows):
            bs = z_0.size(0)
            u, w, b = self.u[k,...].expand(bs, self.z_size, 1), self.w[k,...].expand(bs, 1, self.z_size), self.b[k,...].expand(bs, 1, 1)
            z_k, ldj = self.flow_transformation(z[k], u, w, b)                
            z.append(z_k)
            log_det_jacobian += ldj
        
        return z[-1], log_det_jacobian

    def reverse(self, x):
        raise NotImplementedError("Planar flows cannot be inverted analytically")


class PlanarVAE(VAE):
    """
    Variational auto-encoder with planar flows in the encoder.
    """

    def __init__(self, args):
        super(PlanarVAE, self).__init__(args)
        self.num_flows = args.num_flows
        self.density_evaluation = args.density_evaluation

        # Amortized flow parameters
        if args.density_evaluation:
            # only performing an evaluation of flow, init flow parameters randomly
            self.u = nn.Parameter(torch.randn(self.num_flows, self.z_size, 1).normal_(0, 0.01))
            self.w = nn.Parameter(torch.randn(self.num_flows, 1, self.z_size).normal_(0, 0.01))
            self.b = nn.Parameter(torch.randn(self.num_flows, 1, 1).fill_(0))

        else:
            # u, w, b learned from encoder neural network
            self.amor_u = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)
            self.amor_w = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)
            self.amor_b = nn.Linear(self.q_z_nn_output_dim, self.num_flows)
            self.u, self.w, self.b = None, None, None

        # Normalizing flow layers
        self.flow_transformation = flows.Planar()

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """
        batch_size = x.size(0)

        h = self.q_z_nn(x)
        h = h.view(-1, self.q_z_nn_output_dim)
        z_mu = self.q_z_mean(h)
        z_var = self.q_z_var(h)

        # compute amortized (u, w, b) for flows
        self.u = self.amor_u(h).view(batch_size, self.num_flows, self.z_size, 1)
        self.w = self.amor_w(h).view(batch_size, self.num_flows, 1, self.z_size)
        self.b = self.amor_b(h).view(batch_size, self.num_flows, 1, 1)

        return z_mu, z_var

    def flow(self, z_0):
        # Initialize log-det-jacobian to zero
        log_det_jacobian = 0.0
        z = [z_0]

        for k in range(self.num_flows):
            if self.density_evaluation:
                # Note: it may be faster to not use the batch-wise default transformation in self.flow_transformation()
                # but instead create a non-batch-wise version of that forward step.
                # for now, just expand/repeat the coefficients for each sample
                bs = z_0.size(0)
                u, w, b = self.u[k,...].expand(bs, self.z_size, 1), self.w[k,...].expand(bs, 1, self.z_size), self.b[k,...].expand(bs, 1, 1)
            else:
                u, w, b = self.u[:, k, :, :], self.w[:, k, :, :], self.b[:, k, :, :]

            z_k, ldj = self.flow_transformation(z[k], u, w, b)                
            z.append(z_k)
            log_det_jacobian += ldj
        
        return z[-1], log_det_jacobian

    def forward(self, x):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
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

