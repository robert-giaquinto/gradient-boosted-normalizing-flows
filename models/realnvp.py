import numpy as np
import torch
import torch.nn as nn
import random

from models.vae import VAE
from models.generative_flow import GenerativeFlow
import models.transformations as flows
from models.layers import ReLUNet, ResidualNet, TanhNet, BatchNorm
from utils.utilities import safe_log



class RealNVPFlow(GenerativeFlow):
    """
    RealNVP generative flow model for density estimation
    """
    def __init__(self, args, flip_init=0):
        """
        flip_init: offsets the mask (useful if there are multiple blocks or boosted realnvp components)
                   Can be any positive integer
        """
        super(RealNVPFlow, self).__init__(args)

        self.learn_top = args.learn_top
        self.y_classes = args.y_classes
        self.y_condition = args.y_condition
        self.sample_size = args.sample_size
        self.flip_init = flip_init
        
        #self.flow_step, num_nets = flows.PairedRealNVP(dim=self.z_size), 4
        self.flow_step, num_nets = flows.RealNVP(), 2
        
        self.flow_param = nn.ModuleList()
        for k in range(self.num_flows):
            flow_k = []
            
            flipped = ((k + flip_init) % 2) > 0
            if flipped:
                out_dim = self.z_size // 2
                in_dim = self.z_size - (self.z_size // 2)
            else:
                in_dim = self.z_size // 2
                out_dim = self.z_size - (self.z_size // 2)

            # each realnvp flow step must initialize the 4 coupling networks
            if args.coupling_network == "mixed":
                # scale network s uses TanH, shift network t uses relu
                # this setup is mention in original paper and MAF or MADE paper
                flow_k += [ReLUNet(in_dim, out_dim, args.h_size, args.coupling_network_depth),
                           TanhNet(in_dim, out_dim, args.h_size, args.coupling_network_depth)]
                if num_nets > 2:
                    flow_k += [ReLUNet(out_dim, in_dim, args.h_size, args.coupling_network_depth),
                               TanhNet(out_dim, in_dim, args.h_size, args.coupling_network_depth)]
            else:
                for n in range(num_nets): 
                    if args.coupling_network == "tanh":
                        coupling_network = TanhNet
                    elif args.coupling_network == "residual":
                        coupling_network = ResidualNet
                    elif args.coupling_network == "random":
                        coupling_network = [TanhNet, ReLUNet][np.random.randint(2)]
                    else:
                        coupling_network = ReLUNet

                    if n < 2:
                        flow_k += [coupling_network(in_dim, out_dim, args.h_size, args.coupling_network_depth)]
                    else:
                        flow_k += [coupling_network(out_dim, in_dim, args.h_size, args.coupling_network_depth)]

            if args.batch_norm and k < self.num_flows - 1:
                flow_k += [BatchNorm(self.z_size)]
            else:
                flow_k += [None]

            self.flow_param.append(nn.ModuleList(flow_k))

        self.register_buffer("prior_h", torch.zeros([1, 2 * self.z_size]))
        
    def flow(self, z_0):
        """
        Depracated, just here until I update the density plotting code
        """
        return self.forward(z_0)

    def prior(self, data, y_onehot=None):
        """
        TODO replace with learned prior as in glow
        """
        if data is not None:
            h = self.prior_h.repeat(data.shape[0], 1)
        else:
            h = self.prior_h.repeat(self.sample_size, 1)

        return h[:, :self.z_size], h[:, self.z_size:]

    def decode(self, z, y_onehot, temperature):
        with torch.no_grad():
            if z is None:
                z_mu, z_var = self.prior(z, y_onehot)
                z = torch.normal(z_mu, torch.exp(z_var) * temperature)

            log_det_j = 0.0
            Z = [None for i in range(self.num_flows + 1)]
            Z[-1] = z

            for k in range(self.num_flows, 0, -1):
                flow_k_networks = [self.flow_param[k-1], (k + self.flip_init) % 2]
                z_k, ldj = self.flow_step.inverse(Z[k], flow_k_networks)                
                Z[k-1] = z_k
                log_det_j = log_det_j + ldj

            return Z[0]

    def encode(self, x, y_onehot):
        log_det_j = 0.0
        Z = [x]
        for k in range(self.num_flows):
            flow_k_networks = [self.flow_param[k], (k + self.flip_init) % 2]
            z_k, ldj = self.flow_step(Z[k], flow_k_networks)
            Z.append(z_k)
            log_det_j += ldj

        z_mu, z_var = self.prior(x, y_onehot)
        y_logits = None

        return Z[-1], z_mu, z_var, log_det_j, y_logits

    def forward(self, x=None, y_onehot=None, z=None, temperature=None, reverse=False):
        if reverse:
            return self.decode(z, y_onehot, temperature)
        else:
            return self.encode(x, y_onehot)


class RealNVPVAE(VAE):
    """
    Variational auto-encoder with RealNVP as a flow.
    """
    def __init__(self, args):
        super(RealNVPVAE, self).__init__(args)
        self.num_flows = args.num_flows
        self.density_evaluation = args.density_evaluation
        self.flow_step = flows.RealNVP(dim=self.z_size, use_batch_norm=args.batch_norm)
        
        # Normalizing flow layers
        if args.coupling_network == "relu":
            coupling_network = ReLUNet
        elif args.coupling_network == "residual":
            coupling_network = ResidualNet
        else:
            coupling_network = TanhNet

        in_dim = self.z_size // 2
        #out_dim = self.z_size // 2
        out_dim = self.z_size - (self.z_size // 2)
        self.flow_param = nn.ModuleList()
        for k in range(self.num_flows):
            flow_k = [coupling_network(in_dim, out_dim, args.h_size, args.coupling_network_depth) for _ in range(4)]
            if args.batch_norm: # and k < self.num_flows - 1:
                flow_k += [BatchNorm(self.z_size)]

            self.flow_param.append(nn.ModuleList(flow_k))

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z
        """
        batch_size = x.size(0)
        
        h = self.q_z_nn(x)
        if not self.use_linear_layers:
            h = h.view(h.size(0), -1)
        else:
            h = h.view(-1, self.q_z_nn_output_dim)
            
        z_mu = self.q_z_mean(h)
        z_var = self.q_z_var(h)
        return z_mu, z_var

    def flow(self, z_0):
        log_det_j = 0.0
        Z = [z_0]
        for k in range(self.num_flows):
            flow_k_networks = [self.flow_param[k], k % 2]
            z_k, ldj = self.flow_step(Z[k], flow_k_networks)
            Z.append(z_k)
            log_det_j += ldj
        return Z[-1], log_det_j

    def forward(self, x):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """
        z_mu, z_var = self.encode(x)

        # Sample z_0
        z_0 = self.reparameterize(z_mu, z_var)

        # pass through normalizing flow
        z_k, log_det_j = self.flow(z_0)

        # reconstruct
        x_recon = self.decode(z_k)

        return x_recon, z_mu, z_var, log_det_j, z_0, z_k

