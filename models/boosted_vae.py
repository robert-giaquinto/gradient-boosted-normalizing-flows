import numpy as np
import torch
import torch.nn as nn
import random
import logging

from models.vae import VAE
import models.flows as flows
from optimization.loss import calculate_loss

logger = logging.getLogger(__name__)


class BoostedVAE(VAE):
    """
    Variational auto-encoder with boosted flows.

    """

    def __init__(self, args):
        super(BoostedVAE, self).__init__(args)
        self.args = args
        
        # boosting parameters
        self.num_components = args.num_components
        self.num_flows = args.num_flows
        self.component = 0  # current component being trained / number of components trained thus far
        self.rho = self.FloatTensor(self.num_components).fill_(1.0 / self.num_components)  # mixing weights for components

        # Flow parameters
        if args.component_type == "planar":
            self.flow = flows.Planar()
        else:
            raise ValueError("Lets keep it simple for now and only implement planar components")

        # Amortized flow parameters for each component
        for c in range(self.num_components):
            amor_u = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)
            amor_w = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)
            amor_b = nn.Linear(self.q_z_nn_output_dim, self.num_flows)
            self.add_module('amor_u_' + str(c), amor_u)
            self.add_module('amor_w_' + str(c), amor_w)
            self.add_module('amor_b_' + str(c), amor_b)

    def _rho_gradient(self, x):
        """
        Estimate gradient with Monte Carlo by drawing sample zK ~ g^c and sample zK ~ G^(c-1), and
        computing their densities under the full model G^c
        """
        x = x.detach()
        h, z_mu, z_var = self.encode(x)
        z0 = self.reparameterize(z_mu, z_var)

        # E_g^(c) [ gamma ] sample
        g_zk, _, g_ldj = self.gradient_boosted_flow(h, z0, sample_from="new", density_from="all")
        g_x = self.decode(g_zk)
        gamma_wrt_g, g_recon, g_kl = calculate_loss(g_x, x, z_mu, z_var, z0, g_zk, g_ldj, self.args, beta=1.0)

        # E_G^(c-1) [ gamma ]
        G_zk, _, G_ldj = self.gradient_boosted_flow(h, z0, sample_from="fixed", density_from="all")
        G_x = self.decode(G_zk)
        gamma_wrt_G, G_recon, G_kl = calculate_loss(G_x, x, z_mu, z_var, z0, G_zk, G_ldj, self.args, beta=1.0)

        return gamma_wrt_g, g_recon, g_kl, g_ldj.sum().item(), gamma_wrt_G, G_recon, G_kl, G_ldj.sum().item()
        
    def update_rho(self, data_loader):
        """
        Update rho using equation ___TBD___ with SGD
        """
        if self.component > 0:
            self.eval()
                
            grad_log = open(self.args.snap_dir + '/gradient.log', 'a')
            print('\n\nInitial Rho: ' + ' '.join([f'{val:1.2f}' for val in self.rho.data]), file=grad_log)
            
            step_size = 0.005
            tolerance = 0.0001
            min_iters = 10
            max_iters = 250
            prev_rho = 1.0 / self.num_components

            for batch_id, (x, _) in enumerate(data_loader):
                x.to(self.args.device).detach()

                gamma_wrt_g, g_recon, g_kl, g_ldj, gamma_wrt_G, G_recon, G_kl, G_ldj = self._rho_gradient(x)

                gradient = gamma_wrt_g.detach().item() - gamma_wrt_G.detach().item()
                ss = step_size / (0.01 * batch_id + 1)
                rho = min(max(prev_rho - ss * gradient, 0.025), 1.0)

                grad_msg = f'{batch_id: >3}. rho = {prev_rho:5.3f} -  {gradient:4.2f} * {ss:5.3f} = {rho:5.3f}'
                gamma_msg = f"\tg vs G. Gamma: ({gamma_wrt_g:5.1f}, {gamma_wrt_G:5.1f})."
                gamma_msg += f"\tRecon: ({g_recon:5.1f}, {G_recon:5.1f}).\tKL: ({g_kl:5.1f}, {G_kl:5.1f})."
                gamma_msg += f"\tLDJ: ({g_ldj:4.1f}, {G_ldj:4.1f})"

                print(grad_msg + gamma_msg, file=grad_log)
                
                self.rho[self.component] = rho
                dif = abs(prev_rho - rho)
                prev_rho = rho

                if batch_id > min_iters and (batch_id > max_iters or dif < tolerance):
                    break

            print('New Rho: ' + ' '.join([f'{val:1.2f}' for val in self.rho.data]), file=grad_log)
            grad_log.close()
                
    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """        
        h = self.q_z_nn(x).view(-1, self.q_z_nn_output_dim)
        z_mu = self.q_z_mean(h)
        z_var = self.q_z_var(h)
        return h, z_mu, z_var

    def _sample_component(self, sampling_components):
        if sampling_components == "new":
                j = min(self.component, self.num_components - 1)                
        elif sampling_components in ["all", "fixed"]:
            # sample from either the first c-1 (fixed) or c (fixed + new = all) components
            num_components = min(max(self.component if sampling_components == "fixed" else self.component + 1, 1), self.num_components)
            rho_simplex = self.rho[0:num_components] / torch.sum(self.rho[0:num_components])
            j = torch.multinomial(rho_simplex, 1, replacement=True).item()
        else:
            raise ValueError("z_k can only be sampled from ['new', 'fixed', 'all']")

        return j

    def _get_flow_coefficients(self, c, h):
        batch_size = h.size(0)
        amor_u = getattr(self, 'amor_u_' + str(c))
        amor_w = getattr(self, 'amor_w_' + str(c))
        amor_b = getattr(self, 'amor_b_' + str(c))
        u = amor_u(h).view(batch_size, self.num_flows, self.z_size, 1)
        w = amor_w(h).view(batch_size, self.num_flows, 1, self.z_size)
        b = amor_b(h).view(batch_size, self.num_flows, 1, 1)
        return u, w, b

    def gradient_boosted_flow(self, h, z0, sample_from, density_from):
        batch_size = z0.size(0)

        internal_ldj = self.FloatTensor(batch_size).fill_(0.0)
        external_ldj = self.FloatTensor(batch_size).fill_(0.0)  # also save ldj according to dist that zk is sampled from
        z = [z0]
        
        # which components should zk be sampled from? (i.e. expectation's distribution)
        # Ideally, would sample a different flow for each observation,
        # but instead, for efficiency/simplicity -- just sample a flow PER BATCH
        s = self._sample_component(sampling_components=sample_from)
        u, w, b = self._get_flow_coefficients(s, h)

        if sample_from == density_from:
            # sample zk from component s and compute ldj according to that same component
            for k in range(self.num_flows):
                # sample a z_k and compute density under that same distribution
                zk, ldj = self.flow(z[k], u[:, k, :, :], w[:, k, :, :], b[:, k, :, :])
                z.append(zk)
                internal_ldj += ldj

        else:
            # sample zk from component s, BUT compute ldj according to a DIFFERENT component
            
            # get seperate coefficients to compute density against
            d = self._sample_component(sampling_components=density_from)
            ud, wd, bd = self._get_flow_coefficients(d, h)

            rho_simplex = self.rho[0:self.component + 1] / torch.sum(self.rho[0:self.component + 1])
            with open(self.args.snap_dir + '/sample.log', 'a') as sample_file:
                sample_file.write(f"\nC={self.component: >3}. Sampling {sample_from: <8} selects s={s: <2}, Density {density_from: <8} selects d={d: <2}. Rho: " +
                                  ' '.join([f'{val:1.2f}' for val in rho_simplex.data]))
            
            for k in range(self.num_flows):
                # internal view of sample: sample a z_k and compute ldj
                zk, ildj = self.flow(z[k], u[:, k, :, :], w[:, k, :, :], b[:, k, :, :])
                internal_ldj += ildj
                z.append(zk)
                
                # external view of sample: compute likelihood of z[k] according to "density_from" components
                _, eldj = self.flow(z[k], ud[:, k, :, :], wd[:, k, :, :], bd[:, k, :, :])
                external_ldj += eldj

        return z[-1], internal_ldj, external_ldj

    def forward(self, x, prob_all=0.0):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0 [sum_k log |det dz_k / dz_k-1| ].
        """
        h, z_mu, z_var = self.encode(x)
        z0 = self.reparameterize(z_mu, z_var)
        
        if self.training and self.component < self.num_components:

            if self.training and prob_all > np.random.rand():
                sample_from = 'all'
            else:
                sample_from = 'new'

            # training mode: sample from the new component currently being trained, evaluate it's density according to fixed model
            zk, internal_ldj, external_ldj = self.gradient_boosted_flow(h, z0, sample_from=sample_from, density_from="fixed")
            log_det_jacobian = (internal_ldj, external_ldj)

        else:
            # evaluation mode: sample from any of the first c components
            zk, log_det_jacobian, _ = self.gradient_boosted_flow(h, z0, sample_from="all", density_from="all")
            
            if self.training:
                # all components finished training, sample from all, don't need entropy term
                log_det_jacobian = (None, log_det_jacobian)

        x_recon = self.decode(zk)

        return x_recon, z_mu, z_var, log_det_jacobian, z0, zk
