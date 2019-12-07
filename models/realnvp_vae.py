import numpy as np
import torch
import torch.nn as nn
import random

from models.vae import VAE
import models.flows as flows


class RealNVPVAE(VAE):
    """
    Variational auto-encoder with RealNVP as a flow.
    """

    def __init__(self, args):
        super(RealNVPVAE, self).__init__(args)

        self.h_size = args.h_size
        self.base_network = args.base_network
        self.num_base_layers = args.num_base_layers
        self.num_flows = args.num_flows
        self.density_evaluation = args.density_evaluation
        
        # Normalizing flow layer
        self.flow_transformation = flows.RealNVP(num_flows=self.num_flows,
                                                 dim=self.z_size, hidden_dim=self.h_size,
                                                 base_network=self.base_network,
                                                 num_layers=self.num_base_layers,
                                                 use_batch_norm=False, dropout_probability=0.0)

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z
        """
        batch_size = x.size(0)
        h = self.q_z_nn(x)
        h = h.view(-1, self.q_z_nn_output_dim)
        z_mu = self.q_z_mean(h)
        z_var = self.q_z_var(h)
        return z_mu, z_var

    def flow(self, z_0):
        z_k, log_det_j = self.flow_transformation(z_0)
        return z_k, log_det_j

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

