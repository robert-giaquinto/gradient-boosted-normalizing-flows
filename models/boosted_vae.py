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
        self.density_evaluation = args.density_evaluation
        self.all_trained = False
        
        # boosting parameters
        self.num_components = args.num_components
        self.num_flows = args.num_flows
        self.component = 0  # current component being trained / number of components trained thus far
        self.rho = self.FloatTensor(self.num_components).fill_(1.0 / self.num_components)  # mixing weights for components

        # Flow parameters
        if args.component_type == "planar":
            self.flow_transformation = flows.Planar()
        else:
            raise ValueError("Lets keep it simple for now and only implement planar components")

        if args.density_evaluation:
            self.q_z_nn, self.q_z_mean, self.q_z_var = None, None, None
            self.p_x_nn, self.p_x_mean = None, None
            self.u, self.w, self.b  = nn.ParameterList(), nn.ParameterList(), nn.ParameterList()
        
        # Amortized flow parameters for each component
        for c in range(self.num_components):
            if args.density_evaluation:
                # only performing an evaluation of flow, init flow parameters randomly
                self.u.append(nn.Parameter(torch.randn(self.num_flows, 2, 1).normal_(0, 0.01)))
                self.w.append(nn.Parameter(torch.randn(self.num_flows, 1, 2).normal_(0, 0.01)))
                self.b.append(nn.Parameter(torch.randn(self.num_flows, 1, 1).fill_(0)))
                
            else:
                # u, w, b learned from encoder neural network
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
        g_zk, _, g_ldj = self.flow(z0, sample_from="c", density_from="1:c", h=h)
        g_x = self.decode(g_zk)
        gamma_wrt_g, g_recon, g_kl = calculate_loss(g_x, x, z_mu, z_var, z0, g_zk, g_ldj, self.args, beta=1.0)

        # E_G^(c-1) [ gamma ]
        G_zk, _, G_ldj = self.flow(z0, sample_from="1:c-1", density_from="1:c", h=h)
        G_x = self.decode(G_zk)
        gamma_wrt_G, G_recon, G_kl = calculate_loss(G_x, x, z_mu, z_var, z0, G_zk, G_ldj, self.args, beta=1.0)

        return gamma_wrt_g, g_recon, g_kl, g_ldj.sum().item(), gamma_wrt_G, G_recon, G_kl, G_ldj.sum().item()
        
    def update_rho(self, data_loader):
        """
        Update rho using equation ___TBD___ with SGD
        """
        if self.component > 0:
            self.eval()
            with torch.no_grad():
                
                grad_log = open(self.args.snap_dir + '/gradient.log', 'a')
                print('\n\nInitial Rho: ' + ' '.join([f'{val:1.2f}' for val in self.rho.data]), file=grad_log)
            
                step_size = 0.01
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
        if sampling_components == "c":
            # sample from new component
            j = min(self.component, self.num_components - 1)
            
        elif sampling_components in ["1:c", "1:c-1"]:
            # sample from either the first 1:c-1 (fixed) or 1:c (fixed + new = all) components
            if sampling_components == "1:c-1":
                num_components = self.component
            elif sampling_components == "1:c":
                num_components = self.num_components if self.all_trained else self.component + 1
                
            num_components = min(max(num_components, 1), self.num_components)
            rho_simplex = self.rho[0:num_components] / torch.sum(self.rho[0:num_components])
            j = torch.multinomial(rho_simplex, 1, replacement=True).item()
        elif sampling_components == "-c":
            rho_simplex = self.rho.clone().detach()
            rho_simplex[self.component] = 0.0
            rho_simplex = rho_simplex / rho_simplex.sum()
            j = torch.multinomial(rho_simplex, 1, replacement=True).item()
            
        else:
            raise ValueError("z_k can only be sampled from ['c', '1:c-1', '1:c', '-c'] (corresponding to 'new', 'fixed', or new+fixed components)")

        return j

    def _get_flow_coefficients(self, c, h=None):
        """
        Returns flow coefficients for a particular component c.
        Can compute coefficients based on output of encoder (requires h), or just pull
        the randomly initialized parameters (if doing density evaluation)
        """

        if self.args.density_evaluation:
            u, w, b = self.u[c], self.w[c], self.b[c]
            
        else:
            if h is None:
                raise ValueError("Cannot compute u, w, and b without hidden layer h")
            
            batch_size = h.size(0)
            amor_u = getattr(self, 'amor_u_' + str(c))
            amor_w = getattr(self, 'amor_w_' + str(c))
            amor_b = getattr(self, 'amor_b_' + str(c))
            u = amor_u(h).view(batch_size, self.num_flows, self.z_size, 1)
            w = amor_w(h).view(batch_size, self.num_flows, 1, self.z_size)
            b = amor_b(h).view(batch_size, self.num_flows, 1, 1)
            
        return u, w, b

    def flow(self, z0, sample_from, density_from, h=None):
        bs = z0.size(0)
        entropy_ldj = self.FloatTensor(bs).fill_(0.0)
        boosted_ldj = self.FloatTensor(bs).fill_(0.0)  # also save ldj according to dist that zk is sampled from
        z = [z0]
        
        # which components should zk be sampled from? (i.e. expectation's distribution)
        # Ideally, would sample a different flow for each observation,
        # but instead, for efficiency/simplicity -- just sample a flow PER BATCH
        s = self._sample_component(sampling_components=sample_from)
        u, w, b = self._get_flow_coefficients(s, h=h)

        # get seperate coefficients to compute density against
        # TODO? take multiple samples to compute density against
        d = self._sample_component(sampling_components=density_from)
        ud, wd, bd = self._get_flow_coefficients(d, h=h)

        for k in range(self.num_flows):
            if self.density_evaluation:
                us_k, ws_k, bs_k = u[k,...].expand(bs, 2, 1), w[k,...].expand(bs, 1, 2), b[k,...].expand(bs, 1, 1)
                ud_k, wd_k, bd_k = ud[k,...].expand(bs, 2, 1), wd[k,...].expand(bs, 1, 2), bd[k,...].expand(bs, 1, 1)
            else:
                us_k, ws_k, bs_k = u[:, k, :, :], w[:, k, :, :], b[:, k, :, :]
                ud_k, wd_k, bd_k = ud[:, k, :, :], wd[:, k, :, :], bd[:, k, :, :]
                
            # entropy / internal view of sample: sample a z_k and compute ldj
            zk, eldj = self.flow_transformation(z[k], us_k, ws_k, bs_k)
            entropy_ldj += eldj
            z.append(zk)
            
            # boosted / external view of sample: compute likelihood of z[k] according to "density_from" components
            _, bldj = self.flow_transformation(z[k], ud_k, wd_k, bd_k)
            boosted_ldj += bldj

        return z[-1], entropy_ldj, boosted_ldj

    def forward(self, x, prob_all=0.0):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0 [sum_k log |det dz_k / dz_k-1| ].
        """
        h, z_mu, z_var = self.encode(x)
        z0 = self.reparameterize(z_mu, z_var)
        
        if self.training and self.component < self.num_components:

            if self.training and prob_all > np.random.rand():
                sample_from = '1:c'
                density_from = '1:c'
            else:
                sample_from = 'c'
                density_from = '-c' if self.all_trained else '1:c-1'

            # training mode: sample from the new component currently being trained, evaluate it's density according to fixed model
            zk, entropy_ldj, boosted_ldj = self.flow(z0, sample_from=sample_from, density_from=density_from, h=h)
            log_det_jacobian = (entropy_ldj, boosted_ldj)

        else:
            # evaluation mode: sample from any of the first c components
            # TODO this should returned boosted part of ldj b/c 1:c may shouldn't necesarily sample the same 1:c
            zk, log_det_jacobian, _ = self.flow(z0, sample_from="1:c", density_from="1:c", h=h)
            
            if self.training:
                # all components finished training, sample from all, don't need entropy term
                log_det_jacobian = (None, log_det_jacobian)

        x_recon = self.decode(zk)

        return x_recon, z_mu, z_var, log_det_jacobian, z0, zk
