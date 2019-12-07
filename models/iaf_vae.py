import numpy as np
import torch
import torch.nn as nn
import random

from models.vae import VAE
import models.flows as flows


class IAFVAE(VAE):
    """
    Variational auto-encoder with inverse autoregressive flows in the encoder.
    """

    def __init__(self, args):
        super(IAFVAE, self).__init__(args)
        
        self.h_size = args.h_size
        self.density_evaluation = args.density_evaluation
        
        # flow parameters
        if self.density_evaluation:
            # only performing an evaluation of flow, init flow parameters randomly
            self.h_context = nn.Parameter(torch.randn(self.h_size).normal_(0, 0.01))

        else:
            # learned from encoder neural network per example
            self.amor_h_context = nn.Linear(self.q_z_nn_output_dim, self.h_size)
            self.h_context = None


        # Flow parameters
        self.num_flows = args.num_flows
        self.flow_transformation = flows.IAF(z_size=self.z_size, num_flows=self.num_flows,
                              num_hidden=1, h_size=self.h_size, conv2d=False)

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and context h for flows.
        """

        h = self.q_z_nn(x)
        h = h.view(-1, self.q_z_nn_output_dim)
        mean_z = self.q_z_mean(h)
        var_z = self.q_z_var(h)
        self.h_context = self.amor_h_context(h)

        return mean_z, var_z

    def flow(self, z_0):

        h_context = self.h_context
        if self.density_evaluation:
            batch_size = z_0.size(0)
            h_context = h_context.expand(batch_size, self.h_size)
            
        z_k, log_det_j = self.flow_transformation(z_0, h_context)

        return z_k, log_det_j

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
        z_k, log_det_j = self.flow(z_0, h_context)

        # decode
        x_mean = self.decode(z_k)

        return x_mean, z_mu, z_var, log_det_j, z_0, z_k
