import numpy as np
import torch
import torch.nn as nn
import random

from models.vae import VAE
import models.flows as flows
from models.layers import ReLUNet, ResidualNet


class RealNVPVAE(VAE):
    """
    Variational auto-encoder with RealNVP as a flow.
    """
    def __init__(self, args):
        super(RealNVPVAE, self).__init__(args)
        self.num_flows = args.num_flows
        self.density_evaluation = args.density_evaluation
        self.flow_transformation = flows.RealNVP(dim=self.z_size)
        
        # Normalizing flow layers
        base_network = ResidualNet if args.base_network == "residual" else ReLUNet
        in_dim = self.z_size // 2
        out_dim = self.z_size // 2
        #out_dim = self.z_size - (self.z_size // 2)
        self.flow_param = nn.ModuleList()
        for k in range(self.num_flows):
            flow_param_k = nn.ModuleList([base_network(in_dim, out_dim, args.h_size, args.num_base_layers, use_batch_norm=True) for _ in range(4)])
            self.flow_param.append(flow_param_k)

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
        log_det_jacobian = 0.0
        Z = [z_0]
        for k in range(self.num_flows):
            flow_k_networks = self.flow_param[k]
            z_k, ldj = self.flow_transformation(Z[k], flow_k_networks)
            Z.append(z_k)
            log_det_jacobian += ldj
        return Z[-1], log_det_jacobian

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

