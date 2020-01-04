import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D

import models.flows as flows
from models.layers import GatedConv2d, GatedConvTranspose2d


class VAE(nn.Module):
    """
    Base class for VAE's with normalizing flows.
    """
    def __init__(self, args):
        super(VAE, self).__init__()

        # extract model settings from args
        self.z_size = args.z_size
        self.input_type = args.input_type

        if args.input_size == [1, 28, 20]:
            self.last_kernel_size = (7, 5)
            self.last_pad = 2
        elif args.input_size == [3, 32, 32]:
            self.last_kernel_size = 7
            self.last_pad = 1
        else:
            self.last_kernel_size = 7
            self.last_pad = 2
            
        self.q_z_nn_output_dim = 256
        self.q_z_nn_hidden_dim = 128

        if not args.density_evaluation:
            self.use_linear_layers = args.vae_layers == "linear"
            if self.use_linear_layers:
                self.input_size = np.prod(args.input_size)
            else:
                self.input_size = args.input_size[0]

            self.q_z_nn, self.q_z_mean, self.q_z_var = self.create_encoder()
            self.p_x_nn, self.p_x_mean = self.create_decoder()

        # auxiliary
        if args.cuda:
            self.FloatTensor = torch.cuda.FloatTensor
        else:
            self.FloatTensor = torch.FloatTensor

        # log-det-jacobian = 0 without flows
        self.log_det_j = self.FloatTensor(1).zero_()

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.randn(self.z_size, device=args.device).normal_(0, 0.1))
        #self.register_buffer('base_dist_mean', torch.ones(self.z_size, device=args.device))
        self.register_buffer('base_dist_var', 3.0 * torch.ones(self.z_size, device=args.device))

    @property
    def base_dist(self):
        #rval = D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)
        rval = D.Normal(self.base_dist_mean, self.base_dist_var)
        return rval
    
    def create_encoder(self):
        """
        Helper function to create the elemental blocks for the encoder. Creates a gated convnet encoder.
        the encoder expects data as input of shape (batch_size, num_channels, width, height).
        """
        if self.input_type == 'binary':
            if self.use_linear_layers:
                q_z_nn = nn.Sequential(
                    nn.Linear(self.input_size, self.q_z_nn_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.q_z_nn_hidden_dim, self.q_z_nn_output_dim),
                    nn.Softplus(),
                )
                q_z_mean = nn.Linear(self.q_z_nn_output_dim, self.z_size)
                q_z_var = nn.Sequential(
                    nn.Linear(self.q_z_nn_output_dim, self.z_size),
                    nn.Softplus(),
                )
            else:
                q_z_nn = nn.Sequential(
                    GatedConv2d(self.input_size, 32, 5, 1, 2),
                    GatedConv2d(32, 32, 5, 2, 2),
                    GatedConv2d(32, 64, 5, 1, 2),
                    GatedConv2d(64, 64, 5, 2, self.last_pad),
                    GatedConv2d(64, 64, 5, 1, 2),
                    GatedConv2d(64, self.q_z_nn_output_dim, self.last_kernel_size, 1, 0),
                )
                q_z_mean = nn.Linear(self.q_z_nn_output_dim, self.z_size)
                q_z_var = nn.Sequential(
                    nn.Linear(self.q_z_nn_output_dim, self.z_size),
                    nn.Softplus(),
                )
            return q_z_nn, q_z_mean, q_z_var

        elif self.input_type == 'multinomial':
            act = None
            q_z_nn = nn.Sequential(
                GatedConv2d(self.input_size, 32, 5, 1, 2, activation=act),
                GatedConv2d(32, 32, 5, 2, 2, activation=act),
                GatedConv2d(32, 64, 5, 1, 2, activation=act),
                GatedConv2d(64, 64, 5, 2, self.last_pad, activation=act),
                GatedConv2d(64, 64, 5, 1, 2, activation=act),
                GatedConv2d(64, 256, self.last_kernel_size, 1, 0, activation=act)
            )
            q_z_mean = nn.Linear(256, self.z_size)
            q_z_var = nn.Sequential(
                nn.Linear(256, self.z_size),
                nn.Softplus(),
                nn.Hardtanh(min_val=0.01, max_val=7.)

            )
            return q_z_nn, q_z_mean, q_z_var

    def create_decoder(self):
        """
        Helper function to create the elemental blocks for the decoder. Creates a gated convnet decoder.
        """

        num_classes = 256

        if self.input_type == 'binary':
            if self.use_linear_layers:
                p_x_nn = nn.Sequential(
                    nn.Linear(self.z_size, self.q_z_nn_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.q_z_nn_hidden_dim, self.q_z_nn_output_dim),
                    nn.Softplus(),
                )
                p_x_mean = nn.Sequential(
                    nn.Linear(self.q_z_nn_output_dim, self.input_size),
                    nn.Sigmoid()
                )
            else:
                p_x_nn = nn.Sequential(
                    GatedConvTranspose2d(self.z_size, 64, self.last_kernel_size, 1, 0),
                    GatedConvTranspose2d(64, 64, 5, 1, 2),
                    GatedConvTranspose2d(64, 32, 5, 2, 2, 1),
                    GatedConvTranspose2d(32, 32, 5, 1, 2),
                    GatedConvTranspose2d(32, 32, 5, 2, 2, 1),
                    GatedConvTranspose2d(32, 32, 5, 1, 2)
                )
                p_x_mean = nn.Sequential(
                    nn.Conv2d(32, self.input_size, 1, 1, 0),
                    nn.Sigmoid()
                )
            return p_x_nn, p_x_mean

        elif self.input_type == 'multinomial':
            act = None
            p_x_nn = nn.Sequential(
                GatedConvTranspose2d(self.z_size, 64, self.last_kernel_size, 1, 0, activation=act),
                GatedConvTranspose2d(64, 64, 5, 1, 2, activation=act),
                GatedConvTranspose2d(64, 32, 5, 2, self.last_pad, 1, activation=act),
                GatedConvTranspose2d(32, 32, 5, 1, 2, activation=act),
                GatedConvTranspose2d(32, 32, 5, 2, 2, 1, activation=act),
                GatedConvTranspose2d(32, 32, 5, 1, 2, activation=act)
            )
            p_x_mean = nn.Sequential(
                nn.Conv2d(32, 256, 5, 1, 2),
                nn.Conv2d(256, self.input_size * num_classes, 1, 1, 0),
                # output shape: batch_size, num_channels * num_classes, pixel_width, pixel_height
            )

            return p_x_nn, p_x_mean

        else:
            raise ValueError('invalid input type!!')

    def reparameterize(self, mu, var):
        """
        Samples z from a multivariate Gaussian with diagonal covariance matrix using the
        reparameterization trick.
        """
        std = var.sqrt()
        eps = self.FloatTensor(std.size()).normal_()
        z = eps.mul(std).add_(mu)
        return z

    def encode(self, x):
        """
        Encoder expects following data shapes as input: shape = (batch_size, num_channels, width, height)
        """
        h = self.q_z_nn(x)
        if not self.use_linear_layers:
            h = h.view(h.size(0), -1)
        mean = self.q_z_mean(h)
        var = self.q_z_var(h)
        return mean, var

    def decode(self, z):
        """
        Decoder outputs reconstructed image in the following shapes:
        x_mean.shape = (batch_size, num_channels, width, height)
        """
        if not self.use_linear_layers:
            z = z.view(z.size(0), self.z_size, 1, 1)
        h = self.p_x_nn(z)
        x_mean = self.p_x_mean(h)
        return x_mean

    def forward(self, x):
        """
        Evaluates the model as a whole, encodes and decodes. Note that the log det jacobian is zero
         for a plain VAE (without flows), and z_0 = z_k.
        """
        # mean and variance of z
        z_mu, z_var = self.encode(x)
        # sample z
        z = self.reparameterize(z_mu, z_var)
        x_mean = self.decode(z)
        return x_mean, z_mu, z_var, self.log_det_j, z, z



