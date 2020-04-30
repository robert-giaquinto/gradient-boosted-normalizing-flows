import numpy as np
import torch
import torch.nn as nn
import logging
import torch.distributions as D

from models.generative_flow import GenerativeFlow
from models.glow import Glow
from models.realnvp import RealNVPFlow
from optimization.loss import calculate_loss
from utils.distributions import log_normal_diag, log_normal_standard, log_normal_normalized

logger = logging.getLogger(__name__)


class BoostedFlow(GenerativeFlow):
    """
    Generative flow models trained with gradient boosting
    """

    def __init__(self, args):
        super(BoostedFlow, self).__init__(args)

        self.amortized = not args.density_evaluation
        self.all_trained = False
        self.component_type = args.component_type
        self.num_components = args.num_components
        self.component = 0  # current component being trained / number of components trained thus far

        # Initialize weights rho
        if args.rho_init == "decreasing":
            # each component is given half the weight of the previous one
            self.register_buffer('rho', torch.clamp(
                1.0 / torch.pow(2.0, self.FloatTensor(self.num_components).fill_(0.0) + \
                                torch.arange(self.num_components * 1.0, device=args.device)), min=0.05).to(args.device))
        else:
            # args.rho_init == "uniform"
            #self.register_buffer('rho', self.FloatTensor(self.num_components).fill_(1.0 / self.num_components))
            self.register_buffer('rho', self.FloatTensor(self.num_components).fill_(1.0))

        # initialize component flows
        self.flows = nn.ModuleList()
        for c in range(self.num_components):
            if args.component_type == "realnvp":
                self.flows.append(RealNVPFlow(args, flip_init=c))
            elif args.component_type == "glow":
                self.flows.append(Glow(args))
            else:
                raise NotImplementedError("Only glow and realnvp components are currently implemented")

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
    def _rho_gradient_g(self, x):
        """
        Estimate gradient with Monte Carlo by drawing sample zK ~ g^c and sample zK ~ G^(c-1), and
        computing their densities under the full model G^c
        """
        #x = x.detach().to(self.args.device)
        z_g, mu_g, var_g, ldj_g, _ = self.forward(x=x, components="c")
        g_nll = -1.0 * (log_normal_standard(z_g, reduce=False).view(z_g.shape[0], -1).sum(1, keepdim=False) + ldj_g)

        return g_nll.data.detach()

    @torch.no_grad()
    def _rho_gradient_G(self, x):
        """
        Estimate gradient with Monte Carlo by drawing sample zK ~ g^c and sample zK ~ G^(c-1), and
        computing their densities under the full model G^c
        """
        #x = x.detach().to(self.args.device)
        fixed = "-c" if self.all_trained else "1:c-1"
        z_G, mu_G, var_G, ldj_G, _ = self.forward(x=x, components=fixed)
        G_nll = -1.0 * (log_normal_standard(z_G, reduce=False).view(z_G.shape[0], -1).sum(1, keepdim=False) + ldj_G)
        
        return G_nll.data.detach()
    
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
            num_repeats = self.num_components
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

                g_nll = torch.cat(num_repeats * [self._rho_gradient_g(x)])
                G_nll = torch.cat([self._rho_gradient_G(x) for r in range(num_repeats)])
                sum_ratio = torch.sum(torch.exp(g_nll - G_nll))
                
                gradient = torch.mean(g_nll - G_nll)
                #gradient = torch.mean(G_nll - g_nll)  
                step_size = init_step_size / (0.05 * batch_id + 1)
                rho = min(max(prev_rho - step_size * gradient, 0.01), 100.0)

                grad_msg = f'{batch_id: >3}. rho = {prev_rho:6.4f} -  {gradient:6.3f} * {step_size:7.5f} = {rho:6.4f} '
                loss_msg = f"\tg_nll - G_nll: ({g_nll.mean():6.1f} +/- {g_nll.std():3.1f}, {G_nll.mean():6.1f}  +/- {g_nll.std():3.1f}). Sum g/G={sum_ratio:6.1f} vs n={num_repeats * x.size(0):d}"
                print(grad_msg + loss_msg, file=rho_log)

                self.rho[self.component] = rho
                dif = abs(prev_rho - rho)
                prev_rho = rho

                if batch_id > min_iters and (batch_id > max_iters or dif < tolerance):
                    break

            print('New Rho: ' + ' '.join([f'{val:1.2f}' for val in self.rho.data]), file=rho_log)
            rho_log.close()

    def decode(self, z, y_onehot, temperature, components):
        """
        TODO rewrite this so that samples can be taking from all components,
        not just many samples from a randomly chosen component
        """
        with torch.no_grad():
            c = self._sample_component(components) if type(components) is str else components
            x = self.flows[c](z=z, y_onhot=y_onehot, temperature=temperature, reverse=True)
            
        return x
        
    def encode(self, x, y_onehot, components):
        c = self._sample_component(components) if type(components) is str else components
        return self.flows[c](x)
        
    def forward(self, x=None, y_onehot=None, z=None, temperature=None, components=None, reverse=False):
        if reverse:
            return self.decode(z, y_onehot, temperature, components)
        else:
            return self.encode(x, y_onehot, components)
