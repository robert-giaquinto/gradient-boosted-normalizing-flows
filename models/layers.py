import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F
from utils.distributions import log_normal_diag

from utils.utilities import split_feature, pixels, compute_same_pad, squeeze2d, unsqueeze2d


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
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers=1):
        super().__init__()

        layers = []
        layers += [nn.Linear(in_dim, hidden_dim)]
        layers += []
        for i in range(num_layers):
            layers += [nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)]

        layers += [nn.ReLU(), nn.Linear(hidden_dim, out_dim)]
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class TanhNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers=1):
        super().__init__()

        layers = []
        layers += [nn.Linear(in_dim, hidden_dim)]
        for i in range(num_layers):
            layers += [nn.Tanh(), nn.Linear(hidden_dim, hidden_dim)]

        layers += [nn.Tanh(), nn.Linear(hidden_dim, out_dim)]
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

        self.linear_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(2)
        ])

        if zero_initialization:
            nn.init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            nn.init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs):
        temps = inputs
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[1](temps)
        return inputs + temps


class ResidualNet(nn.Module):
    """
    A general-purpose residual network. Works only with 1-dim inputs.

    TODO: include context features (could be an output of encoder in VAE setup)?
    """
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers=2):
        """
        Note: num_layers refers to the number of residual net blocks (each with 2 linear layers)
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.initial_layer = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(
                hidden_dim=hidden_dim,
            ) for _ in range(num_layers)
        ])
        self.final_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, inputs):
        temps = self.initial_layer(inputs)
        for block in self.blocks:
            temps = block(temps)
        outputs = self.final_layer(temps)
        return outputs


class ConvNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers=1):
        super().__init__()

        layers = []
        layers += [Conv2d(in_dim, hidden_dim)]
        for i in range(num_layers):
            layers += [nn.ReLU(), Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1))]

        layers += [nn.ReLU(), Conv2dZeros(hidden_dim, out_dim)]
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class BatchNorm(nn.Module):
    """
    RealNVP BatchNorm layer
    """
    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer('running_mean', torch.zeros(input_size))
        self.register_buffer('running_var', torch.ones(input_size))
        self.register_buffer('batch_mean', torch.zeros(input_size))
        self.register_buffer('batch_var', torch.zeros(input_size))

    def forward(self, x):
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = x.var(0) # note MAF paper uses biased variance estimate; ie x.var(0, unbiased=False)

            # update running mean
            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # compute normalized input (cf original batch norm paper algo 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta

        # compute log_abs_det_jacobian (cf RealNVP paper)
        log_abs_det_jacobian = self.log_gamma - 0.5 * torch.log(var + self.eps)
        return y, torch.sum(log_abs_det_jacobian.expand_as(x), dim=1)

    def inverse(self, y):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean

        log_abs_det_jacobian = 0.5 * torch.log(var + self.eps) - self.log_gamma
        return x, torch.sum(log_abs_det_jacobian.expand_as(x), dim=1)


class CouplingLayer(nn.Module):
    """
    Modified RealNVP Coupling Layers per the MAF paper

    Performs a single block (flow step) of RealNVP
    """
    def __init__(self, in_dim, out_dim, hidden_dim, mask, num_layers=1, network="tanh", use_batch_norm=True):
        super().__init__()

        if network == "relu":
            base_network = ReLUNet
        elif network == "residual":
            base_network = ResidualNet
        elif network == "random":
            base_network = [TanhNet, ReLUNet][np.random.randint(2)]
        elif network == "tanh":
            base_network = TanhNet

        self.register_buffer('mask', mask)
        self.s_net = base_network(in_dim, out_dim, hidden_dim, num_layers)
        self.t_net = base_network(in_dim, out_dim, hidden_dim, num_layers)

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm = BatchNorm(in_dim)

    def forward(self, x):
        # apply mask
        mx = x * self.mask

        # run through model
        s = self.s_net(mx)
        t = self.t_net(mx)
        y = mx + (1 - self.mask) * (x - t) * torch.exp(-s)  # cf RealNVP eq 8 where y corresponds to x (here we're modeling y)

        log_abs_det_jacobian = torch.sum(-(1 - self.mask) * s, dim=1)  # log det du/dx; cf RealNVP 8 and 6

        if self.use_batch_norm:
            y, ldj = self.batch_norm(y)
            log_abs_det_jacobian = log_abs_det_jacobian + ldj

        return y, log_abs_det_jacobian

    def inverse(self, z):
        # apply mask
        mz = z * self.mask

        # run through model
        s = self.s_net(mz)
        t = self.t_net(mz)
        x = mz + (1 - self.mask) * (z * s.exp() + t)  # cf RealNVP eq 7

        log_abs_det_jacobian = torch.sum((1 - self.mask) * s, dim=1)  # log det dx/du

        if self.use_batch_norm:
            x, ldj = self.batch_norm.inverse(x)
            log_abs_det_jacobian = log_abs_det_jacobian + ldj

        return x, log_abs_det_jacobian



#
#  GLOW LAYERS
#

class _ActNorm(nn.Module):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.
    After initialization, `bias` and `logs` will be trained as parameters.
    """

    def __init__(self, num_features, scale=1.):
        super().__init__()
        # register mean and scale
        size = [1, num_features, 1, 1]
        self.bias = nn.Parameter(torch.zeros(*size))
        self.logs = nn.Parameter(torch.zeros(*size))
        self.num_features = num_features
        self.scale = scale
        self.inited = False

    def initialize_parameters(self, input):
        if not self.training:
            raise ValueError("In Eval mode, but ActNorm not inited")

        with torch.no_grad():
            bias = - torch.mean(input.clone(), dim=[0, 2, 3], keepdim=True)
            vars = torch.mean((input.clone() + bias) ** 2, dim=[0, 2, 3],
                              keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))

            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)

            self.inited = True

    def _center(self, input, reverse=False):
        if reverse:
            return input - self.bias
        else:
            return input + self.bias

    def _scale(self, input, logdet=None, reverse=False):

        if reverse:
            input = input * torch.exp(-self.logs)
        else:
            input = input * torch.exp(self.logs)

        if logdet is not None:
            """
            logs is log_std of `mean of channels`
            so we need to multiply by number of pixels
            """
            b, c, h, w = input.shape

            dlogdet = torch.sum(self.logs) * h * w

            if reverse:
                dlogdet *= -1

            logdet = logdet + dlogdet

        return input, logdet

    def forward(self, input, logdet=None, reverse=False):
        self._check_input_dim(input)

        if not self.inited:
            self.initialize_parameters(input)

        if reverse:
            input, logdet = self._scale(input, logdet, reverse)
            input = self._center(input, reverse)
        else:
            input = self._center(input, reverse)
            input, logdet = self._scale(input, logdet, reverse)

        return input, logdet


class ActNorm2d(_ActNorm):
    def __init__(self, num_features, scale=1.):
        super().__init__(num_features, scale)

    def _check_input_dim(self, input):
        assert len(input.size()) == 4
        assert input.size(1) == self.num_features, (
            "[ActNorm]: input should be in shape as `BCHW`,"
            " channels should be {} rather than {}".format(
                self.num_features, input.size()))


class LinearZeros(nn.Module):
    def __init__(self, in_dim, out_dim, logscale_factor=3):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.logscale_factor = logscale_factor

        self.logs = nn.Parameter(torch.zeros(out_dim))

    def forward(self, input):
        output = self.linear(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class Conv2d(nn.Module):
    def __init__(self, in_dim, out_dim,
                 kernel_size=(3, 3), stride=(1, 1),
                 padding="same", do_actnorm=True, weight_std=0.05):
        super().__init__()

        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride,
                              padding, bias=(not do_actnorm))

        # init weight with std
        self.conv.weight.data.normal_(mean=0.0, std=weight_std)

        if not do_actnorm:
            self.conv.bias.data.zero_()
        else:
            self.actnorm = ActNorm2d(out_dim)

        self.do_actnorm = do_actnorm

    def forward(self, input):
        x = self.conv(input)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x


class Conv2dZeros(nn.Module):
    def __init__(self, in_dim, out_dim,
                 kernel_size=(3, 3), stride=(1, 1),
                 padding="same", logscale_factor=3):
        super().__init__()

        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride,
                              padding)

        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

        self.logscale_factor = logscale_factor
        self.logs = nn.Parameter(torch.zeros(out_dim, 1, 1))

    def forward(self, input):
        output = self.conv(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class Permute2d(nn.Module):
    def __init__(self, num_dim, shuffle):
        super().__init__()
        self.num_dim = num_dim
        self.indices = torch.arange(self.num_dim - 1, -1, -1,
                                    dtype=torch.long)
        self.indices_inverse = torch.zeros((self.num_dim),
                                           dtype=torch.long)

        for i in range(self.num_dim):
            self.indices_inverse[self.indices[i]] = i

        if shuffle:
            self.reset_indices()

    def reset_indices(self):
        shuffle_idx = torch.randperm(self.indices.shape[0])
        self.indices = self.indices[shuffle_idx]

        for i in range(self.num_dim):
            self.indices_inverse[self.indices[i]] = i

    def forward(self, input, reverse=False):
        assert len(input.size()) == 4

        if not reverse:
            input = input[:, self.indices, :, :]
            return input
        else:
            return input[:, self.indices_inverse, :, :]


class Split2d(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv = Conv2dZeros(in_dim // 2, in_dim)

    def split2d_prior(self, z):
        h = self.conv(z)
        return split_feature(h, "cross")

    def forward(self, input, logdet=0.0, reverse=False, temperature=None):
        if reverse:
            z1 = input
            z_mu, z_var = self.split2d_prior(z1)
            z2 = torch.normal(z_mu, torch.exp(z_var) * temperature)
            z = torch.cat((z1, z2), dim=1)
            return z, logdet
        else:
            z1, z2 = split_feature(input, "split")
            z_mu, z_var = self.split2d_prior(z1)
            logdet = log_normal_diag(z2, z_mu, z_var, dim=[1,2,3]) + logdet
            return z1, logdet


class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, logdet=None, reverse=False):
        if reverse:
            output = unsqueeze2d(input, self.factor)
        else:
            output = squeeze2d(input, self.factor)

        return output, logdet


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_dim, LU_decomposed):
        super().__init__()
        w_shape = [num_dim, num_dim]
        w_init = torch.qr(torch.randn(*w_shape))[0]

        if not LU_decomposed:
            self.weight = nn.Parameter(torch.Tensor(w_init))
        else:
            p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
            s = torch.diag(upper)
            sign_s = torch.sign(s)
            log_s = torch.log(torch.abs(s))
            upper = torch.triu(upper, 1)
            l_mask = torch.tril(torch.ones(w_shape), -1)
            eye = torch.eye(*w_shape)

            self.register_buffer('p', p)
            self.register_buffer('sign_s', sign_s)
            self.lower = nn.Parameter(lower)
            self.log_s = nn.Parameter(log_s)
            self.upper = nn.Parameter(upper)
            self.l_mask = l_mask
            self.eye = eye

        self.w_shape = w_shape
        self.LU_decomposed = LU_decomposed

    def get_weight(self, input, reverse):
        b, c, h, w = input.shape

        if not self.LU_decomposed:
            dlogdet = torch.slogdet(self.weight)[1] * h * w
            if reverse:
                weight = torch.inverse(self.weight)
            else:
                weight = self.weight
        else:
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)

            lower = self.lower * self.l_mask + self.eye

            u = self.upper * self.l_mask.transpose(0, 1).contiguous()
            u += torch.diag(self.sign_s * torch.exp(self.log_s))

            dlogdet = torch.sum(self.log_s) * h * w

            if reverse:
                u_inv = torch.inverse(u)
                l_inv = torch.inverse(lower)
                p_inv = torch.inverse(self.p)

                weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))
            else:
                weight = torch.matmul(self.p, torch.matmul(lower, u))

        return weight.view(self.w_shape[0], self.w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)

        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet
