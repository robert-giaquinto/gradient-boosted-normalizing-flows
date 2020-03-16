import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils.utilities import safe_log, split_feature
from models.layers import Conv2d, Conv2dZeros, ActNorm2d, InvertibleConv1x1, Permute2d, LinearZeros, SqueezeLayer, Split2d


class Glow(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.learn_top = args.learn_top
        self.y_classes = args.y_classes
        self.y_condition = args.y_condition
        self.sample_size = args.sample_size
        
        self.flow = FlowNet(image_shape=args.input_size,
                            hidden_dim=args.hidden_dim,
                            K=args.K,
                            L=args.L,
                            actnorm_scale=args.actnorm_scale,
                            flow_permutation=args.flow_permutation,
                            flow_coupling=args.flow_coupling,
                            LU_decomposed=args.LU_decomposed)
        # learned prior
        if self.learn_top:
            C = self.flow.output_shapes[-1][1]
            self.learn_top_fn = Conv2dZeros(C * 2, C * 2)

        if self.y_condition:
            C = self.flow.output_shapes[-1][1]
            self.project_ycond = LinearZeros(self.y_classes, 2 * C)
            self.project_class = LinearZeros(C, self.y_classes)

        self.register_buffer("prior_h",
                             torch.zeros([1,
                                          self.flow.output_shapes[-1][1] * 2,
                                          self.flow.output_shapes[-1][2],
                                          self.flow.output_shapes[-1][3]]))

        # Register bounds to pre-process images, not learnable
        self.register_buffer('bounds', torch.tensor([0.9], dtype=torch.float32))
        
        if args.num_dequant_blocks > 0:
            self.dequant_flows = _Dequantization(in_shape=args.input_size,
                                                 mid_dim=args.dequant_size,
                                                 num_blocks=args.num_dequant_blocks,
                                                 use_attn=args.use_attn,
                                                 drop_prob=args.drop_prob)
        else:
            self.dequant_flows = None

    def prior(self, data, y_onehot=None):
        """
        TODO try replacing with a NICE block, possibly with attention (from MaCOW)
        ??? Should there be a prior at the start of each block like in macow?

        """
        if data is not None:
            h = self.prior_h.repeat(data.shape[0], 1, 1, 1)
        else:
            h = self.prior_h.repeat(self.sample_size, 1, 1, 1)

        dim = h.size(1)

        if self.learn_top:
            h = self.learn_top_fn(h)

        if self.y_condition:
            assert y_onehot is not None
            yp = self.project_ycond(y_onehot)
            h += yp.view(data.shape[0], dim, 1, 1)

        return split_feature(h, "split")

    def forward(self, x=None, y_onehot=None, z=None, temperature=None, reverse=False):
        if reverse:
            return self.decode(z, y_onehot, temperature)
        else:
            return self.encode(x, y_onehot)

    def encode(self, x, y_onehot):
        # Dequant
        logdet = torch.zeros(x.size(0), device=x.device)
        x, logdet = self.dequantize(x, logdet)
        
        # convert to logits
        x, logdet = self.to_logits(x, logdet)

        # apply flow
        z, logdet = self.flow(x, logdet=logdet, reverse=False)
        z_mu, z_var = self.prior(x, y_onehot)

        if self.y_condition:
            y_logits = self.project_class(z.mean(2).mean(2))
        else:
            y_logits = None

        return z, z_mu, z_var, logdet, y_logits

    def decode(self, z, y_onehot, temperature):
        with torch.no_grad():
            if z is None:
                z_mu, z_var = self.prior(z, y_onehot)
                z = torch.normal(z_mu, torch.exp(z_var) * temperature)
                
            x = self.flow(z, temperature=temperature, reverse=True)
            x, _ = self.to_logits(x, 0.0, reverse=True)
        return x

    def dequantize(self, x, accum_logdet):
        if self.dequant_flows is not None:
            x, logdet = self.dequant_flows(x, sldj)
        else:
            # Replace x^i with q^i(x) = U(x, x + 1.0 / 256.0).
            # x: x ~ U(x, x + 1.0 / 256)
            # objective: Equivalent to -q(x)*log(q(x)).
            b, c, h, w = x.size()
            n_bins = 256.0
            chw = c * h * w
            noise = torch.zeros_like(x).uniform_(0, 1.0)
            x = (255.0 * x + noise) / n_bins
            logdet = -math.log(n_bins) * chw * torch.ones(b, device=x.device)

        accum_logdet = accum_logdet + logdet
        return x, accum_logdet

    def to_logits(self, x, accum_logdet, reverse=False):
        """
        Convert the input image `x` to logits.
        Args:
            x (torch.Tensor): Input image.
            accum_logdet (torch.Tensor): Accumulated sum log-determinant of Jacobians.
        Returns:
            y (torch.Tensor): Dequantized logits of `x`.
        See Also:
            - Dequantization: https://arxiv.org/abs/1511.01844, Section 3.1
            - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1
        """
        if reverse:
            x = 1. / (torch.exp(-x) + 1.)    # [0.05, 0.95]
            x *= 2.             # [0.1, 1.9]
            x -= 1.             # [-0.9, 0.9]
            x /= self.bounds    # [-1, 1]
            x += 1.             # [0, 2]
            x /= 2.             # [0, 1]
            return x, 0

        else:
            # restrict data
            x *= 2.             # [0, 2]
            x -= 1.             # [-1, 1]
            x *= self.bounds    # [-0.9, 0.9]
            x += 1.             # [0.1, 1.9]
            x /= 2.             # [0.05, 0.95]
            
            # logit data
            logit_x = torch.log(x) - torch.log(1. - x)

            # Save log-determinant of Jacobian of initial transform
            logdet = F.softplus(logit_x) + F.softplus(-logit_x) \
                - F.softplus((1. - self.bounds).log() - self.bounds.log())
        
            accum_logdet = accum_logdet + logdet.flatten(1).sum(-1)  # better to just sum over dim=[1,2,3]?
            return logit_x, accum_logdet

    def set_actnorm_init(self):
        """
        Use if given a loaded model
        """
        for name, m in self.named_modules():
            if isinstance(m, ActNorm2d):
                m.inited = True


class FlowNet(nn.Module):
    def __init__(self, image_shape, hidden_dim, K, L,
                 actnorm_scale, flow_permutation, flow_coupling,
                 LU_decomposed):
        super().__init__()

        self.layers = nn.ModuleList()
        self.output_shapes = []

        self.K = K
        self.L = L

        C, H, W = image_shape

        for i in range(L):
            # 1. Squeeze
            C, H, W = C * 4, H // 2, W // 2
            self.layers.append(SqueezeLayer(factor=2))
            self.output_shapes.append([-1, C, H, W])

            # 2. K FlowStep
            for _ in range(K):
                self.layers.append(
                    FlowStep(in_dim=C,
                             hidden_dim=hidden_dim,
                             actnorm_scale=actnorm_scale,
                             flow_permutation=flow_permutation,
                             flow_coupling=flow_coupling,
                             LU_decomposed=LU_decomposed))
                self.output_shapes.append([-1, C, H, W])

            # 3. Split2d
            if i < L - 1:
                self.layers.append(Split2d(in_dim=C))
                self.output_shapes.append([-1, C // 2, H, W])
                C = C // 2

    def forward(self, input, logdet=0.0, reverse=False, temperature=None):
        if reverse:
            return self.decode(input, temperature)
        else:
            return self.encode(input, logdet)

    def encode(self, z, logdet=0.0):
        for layer, shape in zip(self.layers, self.output_shapes):
            z, logdet = layer(z, logdet, reverse=False)
        return z, logdet

    def decode(self, z, temperature=None):
        for layer in reversed(self.layers):
            if isinstance(layer, Split2d):
                z, logdet = layer(z, logdet=0, reverse=True, temperature=temperature)
            else:
                z, logdet = layer(z, logdet=0, reverse=True)
        return z



class FlowStep(nn.Module):
    def __init__(self, in_dim, hidden_dim, actnorm_scale,
                 flow_permutation, flow_coupling, LU_decomposed):
        super().__init__()
        self.flow_coupling = flow_coupling

        self.actnorm = ActNorm2d(in_dim, actnorm_scale)

        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = InvertibleConv1x1(in_dim, LU_decomposed=LU_decomposed)
            self.flow_permutation = \
                lambda z, logdet, rev: self.invconv(z, logdet, rev)
        elif flow_permutation == "shuffle":
            self.shuffle = Permute2d(in_dim, shuffle=True)
            self.flow_permutation = \
                lambda z, logdet, rev: (self.shuffle(z, rev), logdet)
        else:
            self.reverse = Permute2d(in_dim, shuffle=False)
            self.flow_permutation = \
                lambda z, logdet, rev: (self.reverse(z, rev), logdet)

        # 3. coupling
        if flow_coupling == "additive":
            self.block = get_block(in_dim // 2,
                                   in_dim // 2,
                                   hidden_dim)
        elif flow_coupling == "affine":
            self.block = get_block(in_dim // 2,
                                   in_dim,
                                   hidden_dim)

    def forward(self, input, logdet=None, reverse=False):
        if reverse:
            return self.decode(input, logdet)
        else:
            return self.encode(input, logdet)


    def encode(self, input, logdet):
        assert input.size(1) % 2 == 0

        # 1. actnorm
        z, logdet = self.actnorm(input, logdet=logdet, reverse=False)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, False)

        # 3. coupling
        z1, z2 = split_feature(z, "split")
        if self.flow_coupling == "additive":
            z2 = z2 + self.block(z1)
        elif self.flow_coupling == "affine":
            h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.)
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        return z, logdet

    def decode(self, input, logdet):
        assert input.size(1) % 2 == 0

        # 1.coupling
        z1, z2 = split_feature(input, "split")
        if self.flow_coupling == "additive":
            z2 = z2 - self.block(z1)
        elif self.flow_coupling == "affine":
            h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.)
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, True)

        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet



class _Dequantization(nn.Module):
    """
    Dequantization Network from Flow++
    
    Args:
        in_shape (int): Shape of the input.
        mid_dim (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
            `Coupling` layers.
        use_attn (bool): Use attention in the coupling layers.
        drop_prob (float): Dropout probability.
        num_flows (int): Number of InvConv+MLCoupling flows to use.
        aux_dim (int): Number of channels in auxiliary input to couplings.
        num_components (int): Number of components in the mixture.
    """
    def __init__(self, in_shape, mid_dim, num_blocks, use_attn, drop_prob,
                 num_flows=4, aux_dim=32, num_components=32):
        super(_Dequantization, self).__init__()
        in_dim, in_height, in_width = in_shape
        self.aux_conv = nn.Sequential(
            nn.Conv2d(2 * in_dim, aux_dim, kernel_size=3, padding=1),
            GatedConv(aux_dim, drop_prob),
            GatedConv(aux_dim, drop_prob),
            GatedConv(aux_dim, drop_prob))

        flows = []
        for _ in range(num_flows):
            flows += [ActNorm(in_dim),
                      InvConv(in_dim),
                      Coupling(in_dim, mid_dim, num_blocks,
                               num_components, drop_prob,
                               use_attn=use_attn,
                               aux_dim=aux_dim),
                      Flip()]
        self.flows = nn.ModuleList(flows)

    def forward(self, x, sldj):
        u = torch.randn_like(x)
        eps_nll = 0.5 * (u ** 2 + math.log(2 * math.pi))

        aux = self.aux_conv(torch.cat(checkerboard(x - 0.5), dim=1))
        u = checkerboard(u)
        for i, flow in enumerate(self.flows):
            u, sldj = flow(u, sldj, aux=aux) if i % 4 == 2 else flow(u, sldj)
        u = checkerboard(u, reverse=True)

        u = torch.sigmoid(u)
        x = (x * 255. + u) / 256.

        sigmoid_ldj = safe_log(u) + safe_log(1. - u)
        sldj = sldj + (eps_nll + sigmoid_ldj).flatten(1).sum(-1)

        return x, sldj
    

def get_block(in_dim, out_dim, hidden_dim):
    block = nn.Sequential(Conv2d(in_dim, hidden_dim),
                          nn.ReLU(inplace=False),
                          Conv2d(hidden_dim, hidden_dim,
                                 kernel_size=(1, 1)),
                          nn.ReLU(inplace=False),
                          Conv2dZeros(hidden_dim, out_dim))
    return block


