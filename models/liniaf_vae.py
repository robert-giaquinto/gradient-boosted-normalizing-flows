import numpy as np
import torch
import torch.nn as nn
import random

from models.vae import VAE
import models.flows as flows


class LinIAFVAE(VAE):
    """
    Variational auto-encoder with linear inverse autoregressive flows in the encoder.
    """

    def __init__(self, args):
        super(LinIAFVAE, self).__init__(args)
        self.num_flows = args.num_flows
        self.density_evaluation = args.density_evaluation

        # Amortized flow parameters
        if args.density_evaluation:
            # only performing an evaluation of flow, init flow parameters randomly
            L = np.repeat(np.eye(self.z_size)[None], self.num_flows, axis=0).astype("float32")
            L[:, 1, 0] = np.random.uniform(low=0.25, high=0.25, size=[self.num_flows]).astype("float32") *\
                np.random.choice([1, -1], size=[self.num_flows])
            self.L = nn.Parameter(torch.from_numpy(L))

        else:
            self.amor_L = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size * self.z_size)
            self.L = None

        # Normalizing flow layers
        self.flow_transformation = flows.LinIAF(self.z_size)

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """
        batch_size = x.size(0)

        h = self.q_z_nn(x)
        h = h.view(-1, self.q_z_nn_output_dim)
        z_mu = self.q_z_mean(h)
        z_var = self.q_z_var(h)

        # compute amortized flow parameter
        self.L = self.amor_L(h).view(batch_size, self.num_flows, self.z_size, self.z_size)
        return z_mu, z_var

    def flow(self, z_0):
        # Initialize log-det-jacobian to zero
        z = [z_0]
        log_det_jacobian = torch.zeros(z_0.size(0))

        for k in range(self.num_flows):
            if self.density_evaluation:
                # Note: it may be faster to not use the batch-wise default transformation in self.flow_transformation()
                # but instead create a non-batch-wise version of that forward step.
                # for now, just expand/repeat the coefficients for each sample
                bs = z_0.size(0)
                L = self.L[k,...].expand(bs, self.z_size, self.z_size)
            else:
                L = self.L[:, k, :, :]

            z_k = self.flow_transformation(z[k], L)
            z.append(z_k)
        
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
