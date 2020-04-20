import numpy as np
import torch
import torch.nn as nn
import random
import torch.distributions as D


class GenerativeFlow(nn.Module):
    """
    Generative flow base class
    For models performing density estimation and matching
    """

    def __init__(self, args):
        super(GenerativeFlow, self).__init__()
        self.num_flows = args.num_flows
        self.z_size = args.z_size
        self.density_evaluation = args.density_evaluation
        self.args = args

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.randn(self.z_size, device=args.device).normal_(0, 0.1))
        self.register_buffer('base_dist_var', 3.0 * torch.ones(self.z_size, device=args.device))

        # Normalizing flow layers
        self.flow_transformation = None

        # auxiliary
        if args.cuda:
            self.FloatTensor = torch.cuda.FloatTensor
        else:
            self.FloatTensor = torch.FloatTensor

        # log-det-jacobian = 0 without flows
        self.log_det_j = self.FloatTensor(1).zero_()


    @property
    def base_dist(self):
        #rval = D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)
        rval = D.Normal(self.base_dist_mean, self.base_dist_var)
        return rval

    def forward(self):
        raise NotImplementedError
