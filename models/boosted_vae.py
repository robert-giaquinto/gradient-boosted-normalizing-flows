import numpy as np
import torch
import torch.nn as nn
import logging

from models.vae import VAE
import models.transformations as flows
from optimization.loss import calculate_loss
from models.layers import ReLUNet, ResidualNet, TanhNet, BatchNorm

logger = logging.getLogger(__name__)


class BoostedVAE(VAE):
    """
    Variational auto-encoder with boosted flows.

    """

    def __init__(self, args):
        super(BoostedVAE, self).__init__(args)

        self.args = args
        self.amortized = not args.density_evaluation
        self.all_trained = False
        
        # boosting parameters
        self.component_type = args.component_type
        self.num_components = args.num_components
        self.num_flows = args.num_flows
        self.component = 0  # current component being trained / number of components trained thus far

        if args.rho_init == "decreasing":
            # each component is given half the weight of the previous one
            self.register_buffer('rho', torch.clamp(
                1.0 / torch.pow(2.0, self.FloatTensor(self.num_components).fill_(0.0) + \
                                torch.arange(self.num_components * 1.0, device=args.device)), min=0.05).to(args.device))
        else:
            # args.rho_init == "uniform"
            self.register_buffer('rho', self.FloatTensor(self.num_components).fill_(1.0 / self.num_components))

        # initialize flow
        if args.component_type == "realnvp":
            self.flow_param = nn.ModuleList()
            self.flow_step = flows.RealNVP(dim=self.z_size, use_batch_norm=args.batch_norm)
        else:
            self.flow_param = nn.ModuleList() if self.amortized else nn.ParameterList()
            if args.component_type == "affine":
                self.flow_step = flows.Affine()
                self.num_coefs = 2
            elif args.component_type == "nlsq":
                self.flow_step = flows.NLSq()
                self.num_coefs = 5
            else:
                raise NotImplementedError("Only affine and nlsq component types are currently implemented")

        # initialize parameters for each component
        for c in range(self.num_components):
            if args.component_type == "realnvp":
                in_dim = self.z_size // 2
                out_dim = self.z_size // 2
                flow_c = nn.ModuleList()
                for k in range(self.num_flows):
                    if args.coupling_network == "mixed":
                        flow_c_k = [ReLUNet(in_dim, out_dim, args.h_size, args.coupling_network_depth),
                                    TanhNet(in_dim, out_dim, args.h_size, args.coupling_network_depth),
                                    ReLUNet(in_dim, out_dim, args.h_size, args.coupling_network_depth),
                                    TanhNet(in_dim, out_dim, args.h_size, args.coupling_network_depth)]
                    else:
                        
                        # realnvp: must initialize the 4 base networks used in each flow (and for each component)
                        flow_c_k = []
                        for _ in range(4):
                            
                            if args.coupling_network == "tanh":
                                coupling_network = TanhNet
                            elif args.coupling_network == "residual":
                                coupling_network = ResidualNet
                            elif args.coupling_network == "random":
                                coupling_network = [TanhNet, ReLUNet][np.random.randint(2)]
                            else:
                                coupling_network = ReLUNet

                            flow_c_k.append(coupling_network(in_dim, out_dim, args.h_size, args.coupling_network_depth))
                            
                        
                    if args.batch_norm and k < self.num_flows - 1:
                        flow_c_k.append(BatchNorm(self.z_size))
                    else:
                        flow_c_k.append(None)
                        
                    flow_c.append(nn.ModuleList(flow_c_k))
                    
                self.flow_param.append(flow_c)

            else:
                # component type is affine or nlsq
                if self.amortized:
                    # amoritize computation of flow parameters, compute coefficients for each datapoint via linear layer
                    flow_param_c = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size * self.num_coefs)
                    self.flow_param.append(flow_param_c)
                else:
                    # flow coefficients NOT data dependent
                    self.flow_param.append(nn.Parameter(torch.randn(self.num_flows, self.z_size, self.num_coefs).normal_(0.0, 0.1)))

    def increment_component(self):
        if self.component == self.num_components - 1:
            # all components have been trained, now loop through and retrain each component
            self.component = 0
            self.all_trained = True
        else:
            # increment to the next component
            self.component = min(self.component + 1, self.num_components - 1)

    def _sample_component(self, sampling_components, batch_size=1):
        """
        Given the argument sampling_components (such as "1:c", "1:c-1", or "-c"), sample a component id from the possible
        components specified by the keyword.

        "1:c":   sample from any of the first c components 
        "1:c-1": sample from any of the first c-1 components 
        "-c":    sample from any component except the c-th component (used during a second pass when fine-tuning components)

        Returns the integer id of the sampled component
        """
        if sampling_components == "c":
            # sample from new component
            j = min(self.component, self.num_components - 1)
            if batch_size > 1:
                j = torch.ones(batch_size) * j
            
        elif sampling_components in ["1:c", "1:c-1"]:
            # sample from either the first 1:c-1 (fixed) or 1:c (fixed + new = all) components
            if sampling_components == "1:c-1":
                num_components = self.component
            elif sampling_components == "1:c":
                num_components = self.num_components if self.all_trained else self.component + 1
                
            num_components = min(max(num_components, 1), self.num_components)
            rho_simplex = self.rho[0:num_components] / torch.sum(self.rho[0:num_components])
            j = torch.multinomial(rho_simplex, batch_size, replacement=True)
            if batch_size == 1:
                j = j.item()
                
        elif sampling_components == "-c":
            rho_simplex = self.rho.clone().detach()
            rho_simplex[self.component] = 0.0
            rho_simplex = rho_simplex / rho_simplex.sum()
            j = torch.multinomial(rho_simplex, batch_size, replacement=True)
            if batch_size == 1:
                j = j.item()
            
        else:
            raise ValueError("z_k can only be sampled from ['c', '1:c-1', '1:c', '-c'] (corresponding to 'new', 'fixed', or new+fixed components)")

        return j

    @torch.no_grad()
    def _rho_gradient(self, x):
        """
        Estimate gradient with Monte Carlo by drawing sample zK ~ g^c and sample zK ~ G^(c-1), and
        computing their densities under the full model G^c
        """
        x = x.detach().to(self.args.device)

        if self.amortized:
            # Must encode and decode the sample
            h, z_mu, z_var = self.encode(x)
            z0 = self.reparameterize(z_mu, z_var)

            # E_g^(c) [ gamma ] sample
            g_zk, _, _, g_ldj = self.flow(z0, sample_from="c", density_from="1:c", h=h)
            g_x = self.decode(g_zk[-1])
            g_loss, _, _ = calculate_loss(g_x, x, z_mu, z_var, z0, g_zk[-1], g_ldj, self.args, beta=1.0)

            # E_G^(c-1) [ gamma ]
            fixed_components = "-c" if self.all_trained else "1:c-1"
            G_zk, _, _, G_ldj = self.flow(z0, sample_from=fixed_components, density_from="1:c", h=h)
            G_x = self.decode(G_zk[-1])
            G_loss, _, _ = calculate_loss(G_x, x, z_mu, z_var, z0, G_zk[-1], G_ldj, self.args, beta=1.0)
        else:
            # No encoding or decoding necessary
            g_zk, _, _, g_ldj = self.flow(x, sample_from="c", density_from="1:c")
            G_zk, _, _, G_ldj = self.flow(x, sample_from="1:c-1", density_from="1:c")

            g_loss = -1.0 * (self.base_dist.log_prob(g_zk[-1]).sum(1) + g_ldj)
            G_loss = -1.0 * (self.base_dist.log_prob(G_zk[-1]).sum(1) + G_ldj)


        return g_loss.mean(0).detach().item(), G_loss.mean(0).detach().item()
        
    def update_rho(self, data_loader):
        """
        Learn weights rho using algorithm and updates from Section 3.3
        """
        if self.component == 0 and self.all_trained == False:
            return

        if self.args.rho_iters == 0:
            return

        self.eval()
        with torch.no_grad():

            rho_log = open(self.args.snap_dir + '/rho.log', 'a')
            print(f"\n\nUpdating weight for component {self.component} (all_trained={str(self.all_trained)})", file=rho_log)
            print('Initial Rho: ' + ' '.join([f'{val:1.2f}' for val in self.rho.data]), file=rho_log)

            tolerance = 0.001
            init_step_size = self.args.rho_lr
            min_iters = 10
            max_iters = self.args.rho_iters
            num_repeats = self.num_components * 3 if self.amortized else 1
            prev_rho = self.rho[self.component].item()

            # create dataloader-iterator
            data_iter = iter(data_loader)
            
            for batch_id in range(max_iters):

                # may need to iterate over the dataset multiple times (in the case of very small test-datasets)
                try:
                    (x, _) = next(data_iter) 
                except StopIteration:
                    data_iter = iter(data_loader)
                    (x, _) = next(data_iter)

                x = x.detach().to(self.args.device)

                if self.amortized:
                    # reshape data for encoder and decoder
                    if self.args.dynamic_binarization:
                        x = torch.bernoulli(x)
                    elif self.args.vae_layers == 'linear':
                        x = x.view(-1, np.prod(self.args.input_size))
                    else:
                        x = x.view(-1, *self.args.input_size)

                g_loss, G_loss = [], []
                for r in range(num_repeats):
                    g, G = self._rho_gradient(x)
                    g_loss.append(g)
                    G_loss.append(G)

                g_loss = np.array(g_loss)
                G_loss = np.array(G_loss)
                gradient = np.mean(g_loss - G_loss)    
                step_size = init_step_size / (0.05 * batch_id + 1)
                rho = min(max(prev_rho - step_size * gradient, 0.0005), 0.999)

                grad_msg = f'{batch_id: >3}. rho = {prev_rho:6.4f} -  {gradient:6.3f} * {step_size:7.5f} = {rho:6.4f} '
                loss_msg = f"\tg vs G. Loss: ({g_loss.mean():6.1f} +/- {g_loss.std():3.1f}, {G_loss.mean():6.1f}  +/- {g_loss.std():3.1f})."
                print(grad_msg + loss_msg, file=rho_log)

                self.rho[self.component] = rho
                dif = abs(prev_rho - rho)
                prev_rho = rho

                if batch_id > min_iters and (batch_id > max_iters or dif < tolerance):
                    break

            print('New Rho: ' + ' '.join([f'{val:1.2f}' for val in self.rho.data]), file=rho_log)
            rho_log.close()

    def _get_flow_parameters(self, c, h=None):
        """
        Returns flow parameters for a particular component c.
        Can compute parameters based on output of encoder (requires h), or just pull
        the randomly initialized parameters (if not amortizing and doing density evaluation)
        """
        if not self.amortized or self.component_type in ["realnvp"]:
            flow_param_c = self.flow_param[c]
            
        else:
            if h is None:
                raise ValueError("Cannot compute flow coefficients without hidden layer h")
            
            batch_size = h.size(0)
            amor_flow_param = self.flow_param[c]
            flow_param_c = amor_flow_param(h).view(batch_size, self.num_flows, self.z_size, self.num_coefs)
            
        return flow_param_c

    def component_forward_flow(self, z_0, component, h=None):
        """
        Extracts the corresponding flow component's parameters, then applies that flow transformation.

        Returns a list of [zc_0, ..., zc_k] and the log det jacobian
        """
        batch_size = z_0.size(0)
        Z = [z_0]

        # get parameters for some component
        flow_param = self._get_flow_parameters(component, h=h)

        # apply forward flow transformation
        log_det_j = self.FloatTensor(batch_size).fill_(0.0)
        for k in range(self.num_flows):
            if self.component_type == "realnvp":
                flow_param_k = [flow_param[k], ((component * self.args.num_flows) + k) % 2]
            else:
                if self.amortized:
                    flow_param_k = flow_param[:, k, :, :]
                else:
                    flow_param_k = flow_param[k,...].expand(batch_size, self.z_size, self.num_coefs)  # repeat coefs for each batch element since there is no encoder to amortize the params

            z_k, ldj = self.flow_step(Z[k], flow_param_k)
            log_det_j = log_det_j + ldj
            Z.append(z_k)

        return Z, log_det_j

    def component_inverse_flow(self, z_K, component, h=None):
        """
        Given a point z_K find the point z_0 using the inverse flow function for a component
        """
        batch_size = z_K.size(0)
        Z = [None for i in range(self.num_flows + 1)]
        Z[-1] = z_K

        # get parameters for some component
        flow_param = self._get_flow_parameters(component, h=h)

        # apply inverse flow transformation
        log_det_j = self.FloatTensor(batch_size).fill_(0.0)
        for k in range(self.num_flows, 0, -1):
            if self.component_type == "realnvp":
                flow_param_k = [flow_param[k-1], ((component * self.args.num_flows) + k) % 2] # second element tells when to flip the ordering
            else:
                if self.amortized:
                    flow_param_k = flow_param[:, k-1, :, :]
                else:
                    flow_param_k = flow_param[k-1, ...].expand(batch_size, self.z_size, self.num_coefs)

            z_k, ldj = self.flow_step.inverse(Z[k], flow_param_k)
            Z[k-1] = z_k
            log_det_j = log_det_j + ldj

        return list(reversed(Z)), log_det_j
                
    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """
        h = self.q_z_nn(x).view(-1, self.q_z_nn_output_dim)
        if self.args.vae_layers == 'linear':
            h = h.view(h.size(0), -1)

        z_mu = self.q_z_mean(h)
        z_var = self.q_z_var(h)
        return h, z_mu, z_var

    def flow(self, z_0, sample_from, density_from=None, h=None):        
        # compute forward step w.r.t. sampling distribution (for training this is the new component g^c)
        # i.e. draw a sample from g(z_K | x) (and same log det jacobian term)
        sample_component = self._sample_component(sampling_components=sample_from)
        zg, entropy_ldj = self.component_forward_flow(z_0, sample_component, h)

        if (self.component == 0 and self.all_trained == False) or density_from is None:
            return zg, entropy_ldj, None, None
        else:
            # compute likelihood of sampled z_K w.r.t fixed components:
            # inverse step: flow_inv(zg[k]) to get zG_0
            self.eval()  # turn off batch-norm updates
            density_component = self._sample_component(sampling_components=density_from)
            zG_0, _ = self.component_inverse_flow(zg[-1], density_component, h)
            
            # compute forward step w.r.t. density distribution (for training this is the fixed components G^(1:c-1))
            zG, boosted_ldj = self.component_forward_flow(zG_0[-1], density_component, h)
            self.train()
            return zg, entropy_ldj, zG, boosted_ldj

    def forward(self, x, prob_all=0.0):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0 [sum_k log |det dz_k / dz_k-1| ].
        """
        h, z_mu, z_var = self.encode(x)
        z_0 = self.reparameterize(z_mu, z_var)

        mix_in_all_components = np.random.rand() < prob_all
        if self.training and not mix_in_all_components:
            # training mode: sample from the new component currently being trained, evaluate it's density according to fixed model
            sample_from = 'c'
            density_from = '-c' if self.all_trained else '1:c-1'
        else:
            # evaluation mode: sample from any of the first c components, only interested in likelihoods w.r.t. that c
            sample_from = '1:c'
            density_from = None

        z_g, g_ldj, z_G, G_ldj = self.flow(z_0, sample_from=sample_from, density_from=density_from, h=h)

        x_recon = self.decode(z_g[-1])
        return x_recon, z_mu, z_var, z_g, g_ldj, z_G, G_ldj
    
