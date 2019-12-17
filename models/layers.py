import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class GatedConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilation=1, activation=None):
        super(GatedConv2d, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)
        self.g = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation(self.h(x))

        g = self.sigmoid(self.g(x))

        return h * g


class GatedConvTranspose2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, output_padding=0, dilation=1,
                 activation=None):
        super(GatedConvTranspose2d, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, output_padding,
                                    dilation=dilation)
        self.g = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, output_padding,
                                    dilation=dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation(self.h(x))

        g = self.sigmoid(self.g(x))

        return h * g


class MaskedLinear(nn.Module):
    """
    Creates masked linear layer for MLP MADE.
    For input (x) to hidden (h) or hidden to hidden layers choose diagonal_zeros = False.
    For hidden to output (y) layers:
    If output depends on input through y_i = f(x_{<i}) set diagonal_zeros = True.
    Else if output depends on input through y_i = f(x_{<=i}) set diagonal_zeros = False.
    """

    def __init__(self, in_features, out_features, diagonal_zeros=False, bias=True):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.diagonal_zeros = diagonal_zeros
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        mask = torch.from_numpy(self.build_mask())
        if torch.cuda.is_available():
            mask = mask.cuda()
        self.mask = torch.autograd.Variable(mask, requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)
        if self.bias is not None:
            self.bias.data.zero_()

    def build_mask(self):
        n_in, n_out = self.in_features, self.out_features
        assert n_in % n_out == 0 or n_out % n_in == 0

        mask = np.ones((n_in, n_out), dtype=np.float32)
        if n_out >= n_in:
            k = n_out // n_in
            for i in range(n_in):
                mask[i + 1:, i * k:(i + 1) * k] = 0
                if self.diagonal_zeros:
                    mask[i:i + 1, i * k:(i + 1) * k] = 0
        else:
            k = n_in // n_out
            for i in range(n_out):
                mask[(i + 1) * k:, i:i + 1] = 0
                if self.diagonal_zeros:
                    mask[i * k:(i + 1) * k:, i:i + 1] = 0
        return mask

    def forward(self, x):
        output = x.mm(self.mask * self.weight)

        if self.bias is not None:
            return output.add(self.bias.expand_as(output))
        else:
            return output

    def __repr__(self):
        if self.bias is not None:
            bias = True
        else:
            bias = False
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ', diagonal_zeros=' \
            + str(self.diagonal_zeros) + ', bias=' \
            + str(bias) + ')'


class MaskedConv2d(nn.Module):
    """
    Creates masked convolutional autoregressive layer for pixelCNN.
    For input (x) to hidden (h) or hidden to hidden layers choose diagonal_zeros = False.
    For hidden to output (y) layers:
    If output depends on input through y_i = f(x_{<i}) set diagonal_zeros = True.
    Else if output depends on input through y_i = f(x_{<=i}) set diagonal_zeros = False.
    """

    def __init__(self, in_features, out_features, size_kernel=(3, 3), diagonal_zeros=False, bias=True):
        super(MaskedConv2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.size_kernel = size_kernel
        self.diagonal_zeros = diagonal_zeros
        self.weight = Parameter(torch.FloatTensor(out_features, in_features, *self.size_kernel))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        mask = torch.from_numpy(self.build_mask())
        if torch.cuda.is_available():
            mask = mask.cuda()
        self.mask = torch.autograd.Variable(mask, requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)
        if self.bias is not None:
            self.bias.data.zero_()

    def build_mask(self):
        n_in, n_out = self.in_features, self.out_features

        assert n_out % n_in == 0 or n_in % n_out == 0, "%d - %d" % (n_in, n_out)

        # Build autoregressive mask
        l = (self.size_kernel[0] - 1) // 2
        m = (self.size_kernel[1] - 1) // 2
        mask = np.ones((n_out, n_in, self.size_kernel[0], self.size_kernel[1]), dtype=np.float32)
        mask[:, :, :l, :] = 0
        mask[:, :, l, :m] = 0

        if n_out >= n_in:
            k = n_out // n_in
            for i in range(n_in):
                mask[i * k:(i + 1) * k, i + 1:, l, m] = 0
                if self.diagonal_zeros:
                    mask[i * k:(i + 1) * k, i:i + 1, l, m] = 0
        else:
            k = n_in // n_out
            for i in range(n_out):
                mask[i:i + 1, (i + 1) * k:, l, m] = 0
                if self.diagonal_zeros:
                    mask[i:i + 1, i * k:(i + 1) * k:, l, m] = 0

        return mask

    def forward(self, x):
        output = F.conv2d(x, self.mask * self.weight, bias=self.bias, padding=(1, 1))
        return output

    def __repr__(self):
        if self.bias is not None:
            bias = True
        else:
            bias = False
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ', diagonal_zeros=' \
            + str(self.diagonal_zeros) + ', bias=' \
            + str(bias) + ', size_kernel=' \
            + str(self.size_kernel) + ')'


class ReLUNet(nn.Module):
    """
    Simple fully connected neural network with ReLU activations.

    TODO: change this to a Fully Connected Network with a activation passed as an argument
    """
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers=1, use_batch_norm=False):
        super().__init__()

        layers = []
        layers += [nn.Linear(in_dim, hidden_dim)]
        layers += [nn.ReLU()]
        for i in range(num_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim)]
            layers += [nn.ReLU()]
            if use_batch_norm:
                layers += [nn.BatchNorm1d(hidden_dim)]

        layers += [nn.Linear(hidden_dim, out_dim)]
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class ResidualBlock(nn.Module):
    """
    A general-purpose residual block. Works only with 1-dim inputs.

    TODO: allow fo a passed activation function
    """
    def __init__(self, hidden_dim, use_batch_norm=True, zero_initialization=True):
        super().__init__()
        
        self.activation = nn.ReLU()
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim, eps=1e-3)
                for _ in range(2)
            ])

        self.linear_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(2)
        ])

        if zero_initialization:
            nn.init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            nn.init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[1](temps)
        return inputs + temps


class ResidualNet(nn.Module):
    """
    A general-purpose residual network. Works only with 1-dim inputs.

    TODO: include context features (could be an output of encoder in VAE setup)?
    """
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers=2, use_batch_norm=True):
        """
        Note: num_layers refers to the number of residual net blocks (each with 2 linear layers)
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.initial_layer = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(
                hidden_dim=hidden_dim,
                use_batch_norm=use_batch_norm,
            ) for _ in range(num_layers)
        ])
        self.final_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, inputs):
        temps = self.initial_layer(inputs)
        for block in self.blocks:
            temps = block(temps)
        outputs = self.final_layer(temps)
        return outputs
