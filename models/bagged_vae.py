import numpy as np
import torch
import torch.nn as nn
import random
import logging

from models.vae import VAE
import models.flows as flows

logger = logging.getLogger(__name__)


class BaggedVAE(VAE):
    """
    Variational auto-encoder with bagged planar flows in the encoder.
    """
    def __init__(self, args):
        super(BaggedVAE, self).__init__(args)

        # bagging parameters
        self.component_type = args.component_type
        self.num_components = args.num_components
        self.component = 0  # last component chosen during training
        self.num_flows = args.num_flows

        # mixing weights for components
        self.rho = self.FloatTensor(self.num_components).fill_(1.0).detach()
        self.rho += 0.1 * self.FloatTensor(self.num_components).normal_()
        self.rho = self.rho / self.rho.sum()

        # Initialize log-det-jacobian to zero
        self.log_det_j = self.FloatTensor(1, 1).fill_(0.0)

        
        self.q_z_nn = nn.Sequential(
            nn.Linear(np.prod(self.input_size), self.q_z_nn_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.q_z_nn_hidden_dim, self.q_z_nn_output_dim),
            nn.Softplus(),
        )

        for c in range(self.num_components):
            q_z_mean = nn.Linear(self.q_z_nn_output_dim, self.z_size)
            q_z_var = nn.Sequential(
                nn.Linear(self.q_z_nn_output_dim, self.z_size),
                nn.Softplus())
            self.add_module('q_z_mean_' + str(c), q_z_mean)
            self.add_module('q_z_var_' + str(c), q_z_var)

        self.q_z_mean = None
        self.q_z_var = None

        # Flow parameters
        if self.component_type == "planar":
            self.flow_transformation = flows.Planar()
            # Amortized flow parameters for each weak component
            for c in range(self.num_components):
                amor_u = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)
                amor_w = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)
                amor_b = nn.Linear(self.q_z_nn_output_dim, self.num_flows)
                self.add_module('amor_u_' + str(c), amor_u)
                self.add_module('amor_w_' + str(c), amor_w)
                self.add_module('amor_b_' + str(c), amor_b)

        elif self.component_type == "radial":
            self.flow_transformation = flows.Radial
            for c in range(self.num_components):
                amor_alpha = nn.Sequential(
                    nn.Linear(self.q_z_nn_output_dim, self.num_flows),
                    nn.Softplus(),
                    nn.Hardtanh(min_val=0.01, max_val=7.)
                )
                amor_beta = nn.Linear(self.q_z_nn_output_dim, self.num_flows)
                amor_zref = nn.Linear(self.q_z_nn_output_dim, self.z_size)
                self.add_module('amor_a_' + str(c), amor_alpha)
                self.add_module('amor_b_' + str(c), amor_beta)
                self.add_module('amor_z_' + str(c), amor_zref)

        else:
            raise ValueError("Only radial or planar weak components allowed for now.")

    def sample_component(self, batch_id):
        # only allow the component to be sampled from 2/3 of the data
        rho_hat = self.rho + 0.001
        if batch_id is not None:
            rho_hat = torch.FloatTensor([r if c % 3 != batch_id % 3 else 0.0 for c, r in enumerate(rho_hat)])

        # normalize rho so values representation probability
        rho_simplex = rho_hat / torch.sum(rho_hat)

        # sample a single component
        component = torch.multinomial(rho_simplex, 1, replacement=True).item()
        return component

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """
        batch_size = x.size(0)

        h = self.q_z_nn(x)
        h = h.view(-1, self.q_z_nn_output_dim)

        q_z_mean = getattr(self, 'q_z_mean_' + str(self.component))
        q_z_var = getattr(self, 'q_z_var_' + str(self.component))
        z_mu = q_z_mean(h)
        z_var = q_z_var(h)

        # return amortized flow parameters for all flows

        if self.component_type == "planar":
            amor_u = getattr(self, 'amor_u_' + str(self.component))
            amor_w = getattr(self, 'amor_w_' + str(self.component))
            amor_b = getattr(self, 'amor_b_' + str(self.component))
            u = amor_u(h).view(batch_size, self.num_flows, self.z_size, 1)
            w = amor_w(h).view(batch_size, self.num_flows, 1, self.z_size)
            b = amor_b(h).view(batch_size, self.num_flows, 1, 1)
            flow_params = [u, w, b]

        elif self.component_type == "radial":
            amor_alpha = getattr(self, 'amor_a_' + str(self.component))
            amor_beta = getattr(self, 'amor_b_' + str(self.component))
            amor_zref = getattr(self, 'amor_z_' + str(self.component))
            alpha = amor_alpha(h).view(batch_size, self.num_flows, 1, 1)
            beta = amor_beta(h).view(batch_size, self.num_flows, 1, 1)
            z_ref = amor_zref(h).view(batch_size, self.z_size)
            flow_params = [alpha, beta, z_ref]

        else:
            raise ValueError("Only radial or planar weak components allowed for now.")

        return z_mu, z_var, flow_params

    def flow(self, z_0, flow_params):
        log_det_jacobian = self.FloatTensor(z_0.size(0)).fill_(0.0)
        Z_arr = [z_0]
        
        # apply flow transformations
        for k in range(self.num_flows):
            if self.component_type == "planar":
                u, w, b = flow_params
                z_k, ldj = self.flow_transformation(Z_arr[k], u[:, k, :, :], w[:, k, :, :], b[:, k, :, :])
                
            elif self.component_type == "radial":
                alpha, beta, z_ref = flow_params
                z_k, ldj = self.flow_transformation(Z_arr[k], z_ref, alpha[:, k, :, :], beta[:, k, :, :])

            Z_arr.append(z_k)
            log_det_jacobian += ldj

        z_out = Z_arr[-1]
        return z_out, log_det_jacobian


    def forward(self, x, batch_id=None):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0 [sum_k log |det dz_k / dz_k-1| ].
        """
        # Normalizing flows selection
        self.component = self.sample_component(batch_id)
        
        z_mu, z_var, flow_params = self.encode(x)
        
        z_0 = self.reparameterize(z_mu, z_var)

        z_k, log_det_jacobian = self.flow(z_0, flow_params)
        
        # decode aggregated output of weak components
        x_mean = self.decode(z_k)

        return x_mean, z_mu, z_var, log_det_jacobian, z_0, z_out

    def update_rho(self, component_loss):
        """
        maximization-step, update rho based on total observed component losses
        """
        if component_loss.min() < 0.0:
            raise ValueError("Component losses are less than zero, that shouldn't happen probably. Maybe add min here?")

        component_loss[1, :] += 1.0 # for numerical stability
        component_loss = component_loss[0, :] / component_loss[1, :]
        
        loss_recipricol = 1.0 / component_loss
        old_rho = self.rho
        self.rho = torch.from_numpy(np.exp(loss_recipricol) / np.sum(np.exp(loss_recipricol))).float().detach()
        self.rho += 0.01 * self.FloatTensor(self.num_components).normal_()
        self.rho = self.rho / self.rho.sum()
