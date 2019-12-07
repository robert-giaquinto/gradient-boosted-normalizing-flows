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
        if args.component_type == "affine":
            self.flow_transformation = flows.Affine()
            self.num_coefs = 2
        elif args.component_type == "nlsq":
            self.flow_transformation = flows.NLSq()
            self.num_coefs = 5
        elif args.component_type == "realnvp":
            self.base_network = args.base_network
            self.h_size = args.h_size
            self.num_base_layers = args.num_base_layers
            self.flow_transformation == flows.RealNVP(num_flows=self.num_flows,
                                                      dim=self.z_size, hidden_dim=self.h_size,
                                                      base_network=self.base_network,
                                                      num_layers=self.num_base_layers,
                                                      use_batch_norm=False, dropout_probability=0.0)
        else:
            raise NotImplementedError("Only affine and nlsq component types are currently implemented")

        if args.density_evaluation:
            self.q_z_nn, self.q_z_mean, self.q_z_var = None, None, None
            self.p_x_nn, self.p_x_mean = None, None
            self.flow_coef = nn.ParameterList()

        # Amortized flow parameters for each component
        if args.component_type in ['affine', 'nlsq']:
            for c in range(self.num_components):
                if args.density_evaluation:
                    self.flow_coef.append(nn.Parameter(torch.randn(self.num_flows, self.z_size, self.num_coefs).normal_(0.0, 0.1)))
                else:
                    amor_flow_coef = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size * self.num_coefs)
                    self.add_module('amor_flow_coef_' + str(c), amor_flow_coef)

    def increment_component(self):
        if self.component == self.num_components - 1:
            # all components have been trained, now loop through and retrain each component
            self.component = 0
            self.all_trained = True
        else:
            # increment to the next component
            self.component = min(self.component + 1, self.num_components - 1)

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
        """
        Given the keyword "sampling_components" (such as "1:c", "1:c-1", or "-c"), sample a component id from the possible
        components specified by the keyword.

        "1:c":   sample from any of the first c components
        "1:c-1": sample from any of the first c-1 components
        "-c":    sample from any component except the c-th component

        Returns the integer id of the sampled component
        """
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
            flow_coef = self.flow_coef[c]
            
        else:
            if h is None:
                raise ValueError("Cannot compute flow coefficients without hidden layer h")
            
            batch_size = h.size(0)
            amor_flow_coef = getattr(self, 'amor_flow_coef_' + str(c))
            flow_coef = amor_flow_coef(h).view(batch_size, self.num_flows, self.z_size, self.num_coefs)
            
        return flow_coef

    def component_forward_flow(self, z0, component, h=None):
        """
        Get the corresponding flow's coefficients, then apply the flow

        Returns a list of [z0, ..., zk] and the log det jacobian
        """
        bs = z0.size(0)
        Z = [z0]
            
        flow_coef = self._get_flow_coefficients(component, h=h)

        log_det_j = self.FloatTensor(bs).fill_(0.0)
        for k in range(self.num_flows):
            if self.density_evaluation:
                flow_coef_k = flow_coef[k,...].expand(bs, self.z_size, self.num_coefs)
            else:
                flow_coef_k = flow_coef[:, k, :, :]

            zk, ldj = self.flow_transformation(Z[k], flow_coef_k)
            log_det_j += ldj
            Z.append(zk)

        return Z, log_det_j

    def component_inverse_flow(self, z_K, component, h=None):
        """
        Given a point z_K find the point z0 using the inverse flow function for a component
        """
        bs = z_K.size(0)
        Z = [None for i in range(self.num_flows + 1)]
        Z[-1] = z_K

        flow_coef = self._get_flow_coefficients(component, h=h)
        for k in range(self.num_flows, 0, -1):
            if self.density_evaluation:
                flow_coef_k = flow_coef[k-1, ...].expand(bs, self.z_size, self.num_coefs)
            else:
                flow_coef_k = flow_coef[:, k-1, :, :]

            z_k, _ = self.flow_transformation.inverse(Z[k], flow_coef_k)
            Z[k-1] = z_k

        return Z[0]

    def flow(self, z0, sample_from, density_from, h=None):        
        # compute forward step w.r.t. sampling distribution (for training this is the new component g^c)
        # i.e. draw a sample from g(z_K | x) (and same log det jacobian term)
        sample_component = self._sample_component(sampling_components=sample_from)
        z_g, entropy_ldj = self.component_forward_flow(z0, sample_component, h)

        if self.component == 0 and self.all_trained == False:
            return z_g, entropy_ldj, None, None
        else:
            # compute likelihood of sampled z_K w.r.t fixed components:
            # inverse step: flow_inv(z_g[k]) to get z_0^G
            density_component = self._sample_component(sampling_components=density_from)
            z0_G = self.component_inverse_flow(z_g[-1], density_component, h)
            
            # compute forward step w.r.t. density distribution (for training this is the fixed components G^(1:c-1))
            z_G, boosted_ldj = self.component_forward_flow(z0_G, density_component, h)
                
            return z_g, entropy_ldj, z_G, boosted_ldj

    def forward(self, x):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0 [sum_k log |det dz_k / dz_k-1| ].
        """
        h, z_mu, z_var = self.encode(x)
        z0 = self.reparameterize(z_mu, z_var)
        
        if self.training and self.component < self.num_components:
            # training mode: sample from the new component currently being trained, evaluate it's density according to fixed model
            sample_from = 'c'
            density_from = '-c' if self.all_trained else '1:c-1'

            # use this if we don't implement multiple decoders: with prob_all init at 0.0
            #if self.training and prob_all > np.random.rand():
            #    sample_from = '1:c'
            #    density_from = '1:c'

        else:
            # evaluation mode: sample from any of the first c components
            sample_from = "1:c"
            density_from = "1:c"

        z_g, entropy_ldj, z_G, boosted_ldj = self.flow(z0, sample_from=sample_from, density_from=density_from, h=h)

        x_recon = self.decode(z_g[-1])

        return x_recon, z_mu, z_var, z_g, entropy_ldj, z_G, boosted_ldj
    
