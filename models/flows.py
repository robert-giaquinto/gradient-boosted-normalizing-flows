import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils.utilities import safe_log
from models.layers import MaskedConv2d, MaskedLinear


class Planar(nn.Module):
    """
    PyTorch implementation of planar flows as presented in "Variational Inference with Normalizing Flows"
    by Danilo Jimenez Rezende, Shakir Mohamed. Model assumes amortized flow parameters.
    """

    def __init__(self):
        super(Planar, self).__init__()
        self.h = nn.Tanh()
        self.softplus = nn.Softplus()

    def der_h(self, x):
        """ Derivative of tanh """
        return 1 - self.h(x) ** 2

    def transform(self, zk, u, w, b):
        """
        non batched version of the forward step (may be faster for density estimation)
        """
        #zk = zk.unsqueeze(2)

        # reparameterize u such that the flow becomes invertible (see appendix paper)
        uw = w @ u
        m_uw = -1. + self.softplus(uw)
        #w_norm_sq = torch.sum(w ** 2, dim=2, keepdim=True)
        w_norm_sq = w @ w.t()
        u_hat = u + ((m_uw - uw) * w.t() / w_norm_sq)

        # compute flow with u_hat
        wzb = zk @ w.t() + b
        z = zk + u_hat.t() * self.h(wzb)

        # compute logdetJ
        psi = w * self.der_h(wzb)
        log_det_jacobian = safe_log(torch.abs(1 + psi @ u_hat))
        log_det_jacobian = log_det_jacobian.squeeze()

        return z, log_det_jacobian        
        
    def forward(self, zk, z0, u, w, b):
        """
        Forward pass. Assumes amortized u, w and b. Conditions on diagonals of u and w for invertibility
        will be be satisfied inside this function. Computes the following transformation:
        z' = z + u h( w^T z + b)
        or actually
        z'^T = z^T + h(z^T w + b)u^T
        Assumes the following input shapes:
        shape u = (batch_size, z_size, 1)
        shape w = (batch_size, 1, z_size)
        shape b = (batch_size, 1, 1)
        shape z = (batch_size, z_size).
        """
        zk = zk.unsqueeze(2)

        # reparameterize u such that the flow becomes invertible (see appendix paper)
        uw = torch.bmm(w, u)
        m_uw = -1. + self.softplus(uw)
        w_norm_sq = torch.sum(w ** 2, dim=2, keepdim=True)
        u_hat = u + ((m_uw - uw) * w.transpose(2, 1) / w_norm_sq)

        # compute flow with u_hat
        wzb = torch.bmm(w, z0) + b
        z = zk + u_hat * self.h(wzb)
        z = z.squeeze(2)

        # compute logdetJ
        psi = w * self.der_h(wzb)
        log_det_jacobian = safe_log(torch.abs(1 + torch.bmm(psi, u_hat)))
        log_det_jacobian = log_det_jacobian.squeeze(2).squeeze(1)

        return z, log_det_jacobian


class Radial(nn.Module):
    """
    PyTorch implementation of radial flows as presented in "Variational Inference with Normalizing Flows"
    by Danilo Jimenez Rezende, Shakir Mohamed. Model assumes amortized flow parameters.
    """

    def __init__(self):

        super(Radial, self).__init__()
        self.softplus = nn.Softplus()

    def der_h(self, r, alpha):
        """
        Derivative of h function in paper (from eq 14)
        """
        return -1.0 / ((alpha + r)**2)

    def forward(self, zk, z0, alpha, beta):
        """
        Forward pass.
        """
        z_size = z0.size(1)
        zk = zk.unsqueeze(2)
        z0 = z0.unsqueeze(2)

        # reparameterize u such that the flow becomes invertible (see appendix paper)
        zk_z0 = zk - z0
        r = torch.norm(zk_z0, dim=1).unsqueeze(1)
        h = 1.0 / (alpha + r)
        beta_hat = -alpha * self.softplus(beta)
        beta_h = beta_hat * h
        z = zk + beta_h * zk_z0
        z = z.squeeze(2)

        # compute logdetJ
        log_det_jacobian = safe_log(torch.bmm(1 + beta_h + (beta_hat * self.der_h(r, alpha) * r),
            (1.0 + beta_h)**(z_size - 1)))
        log_det_jacobian = log_det_jacobian.squeeze(2).squeeze(1)

        return z, log_det_jacobian


class Sylvester(nn.Module):
    """
    Sylvester normalizing flow.
    """

    def __init__(self, num_ortho_vecs):

        super(Sylvester, self).__init__()

        self.num_ortho_vecs = num_ortho_vecs

        self.h = nn.Tanh()

        triu_mask = torch.triu(torch.ones(num_ortho_vecs, num_ortho_vecs), diagonal=1).unsqueeze(0)
        diag_idx = torch.arange(0, num_ortho_vecs).long()

        self.register_buffer('triu_mask', Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer('diag_idx', diag_idx)

    def der_h(self, x):
        return self.der_tanh(x)

    def der_tanh(self, x):
        return 1 - self.h(x) ** 2

    def _forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True):
        """
        All flow parameters are amortized. Conditions on diagonals of R1 and R2 for invertibility need to be satisfied
        outside of this function. Computes the following transformation:
        z' = z + QR1 h( R2Q^T z + b)
        or actually
        z'^T = z^T + h(z^T Q R2^T + b^T)R1^T Q^T
        :param zk: shape: (batch_size, z_size)
        :param r1: shape: (batch_size, num_ortho_vecs, num_ortho_vecs)
        :param r2: shape: (batch_size, num_ortho_vecs, num_ortho_vecs)
        :param q_ortho: shape (batch_size, z_size , num_ortho_vecs)
        :param b: shape: (batch_size, 1, self.z_size)
        :return: z, log_det_j
        """

        # Amortized flow parameters
        zk = zk.unsqueeze(1)

        # Save diagonals for log_det_j
        diag_r1 = r1[:, self.diag_idx, self.diag_idx]
        diag_r2 = r2[:, self.diag_idx, self.diag_idx]

        r1_hat = r1
        r2_hat = r2

        qr2 = torch.bmm(q_ortho, r2_hat.transpose(2, 1))
        qr1 = torch.bmm(q_ortho, r1_hat)

        r2qzb = torch.bmm(zk, qr2) + b
        z = torch.bmm(self.h(r2qzb), qr1.transpose(2, 1)) + zk
        z = z.squeeze(1)

        # Compute log|det J|
        # Output log_det_j in shape (batch_size) instead of (batch_size,1)
        diag_j = diag_r1 * diag_r2
        diag_j = self.der_h(r2qzb).squeeze(1) * diag_j
        diag_j += 1.
        log_diag_j = safe_log(diag_j.abs())

        if sum_ldj:
            log_det_j = log_diag_j.sum(-1)
        else:
            log_det_j = log_diag_j

        return z, log_det_j

    def forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True):

        return self._forward(zk, r1, r2, q_ortho, b, sum_ldj)


class TriangularSylvester(nn.Module):
    """
    Sylvester normalizing flow with Q=P or Q=I.
    """

    def __init__(self, z_size):

        super(TriangularSylvester, self).__init__()

        self.z_size = z_size
        self.h = nn.Tanh()

        diag_idx = torch.arange(0, z_size).long()
        self.register_buffer('diag_idx', diag_idx)

    def der_h(self, x):
        return self.der_tanh(x)

    def der_tanh(self, x):
        return 1 - self.h(x) ** 2

    def _forward(self, zk, r1, r2, b, permute_z=None, sum_ldj=True):
        """
        All flow parameters are amortized. conditions on diagonals of R1 and R2 need to be satisfied
        outside of this function.
        Computes the following transformation:
        z' = z + QR1 h( R2Q^T z + b)
        or actually
        z'^T = z^T + h(z^T Q R2^T + b^T)R1^T Q^T
        with Q = P a permutation matrix (equal to identity matrix if permute_z=None)
        :param zk: shape: (batch_size, z_size)
        :param r1: shape: (batch_size, num_ortho_vecs, num_ortho_vecs).
        :param r2: shape: (batch_size, num_ortho_vecs, num_ortho_vecs).
        :param b: shape: (batch_size, 1, self.z_size)
        :return: z, log_det_j
        """

        # Amortized flow parameters
        zk = zk.unsqueeze(1)

        # Save diagonals for log_det_j
        diag_r1 = r1[:, self.diag_idx, self.diag_idx]
        diag_r2 = r2[:, self.diag_idx, self.diag_idx]

        if permute_z is not None:
            # permute order of z
            z_per = zk[:, :, permute_z]
        else:
            z_per = zk

        r2qzb = torch.bmm(z_per, r2.transpose(2, 1)) + b
        z = torch.bmm(self.h(r2qzb), r1.transpose(2, 1))

        if permute_z is not None:
            # permute order of z again back again
            z = z[:, :, permute_z]

        z += zk
        z = z.squeeze(1)

        # Compute log|det J|
        # Output log_det_j in shape (batch_size) instead of (batch_size,1)
        diag_j = diag_r1 * diag_r2
        diag_j = self.der_h(r2qzb).squeeze(1) * diag_j
        diag_j += 1.
        log_diag_j = safe_log(diag_j.abs())

        if sum_ldj:
            log_det_j = log_diag_j.sum(-1)
        else:
            log_det_j = log_diag_j

        return z, log_det_j

    def forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True):

        return self._forward(zk, r1, r2, q_ortho, b, sum_ldj)


class IAF(nn.Module):
    """
    PyTorch implementation of inverse autoregressive flows as presented in
    "Improving Variational Inference with Inverse Autoregressive Flow" by Diederik P. Kingma, Tim Salimans,
    Rafal Jozefowicz, Xi Chen, Ilya Sutskever, Max Welling.
    Inverse Autoregressive Flow with either MADE MLPs or Pixel CNNs. Contains several flows. Each transformation
     takes as an input the previous stochastic z, and a context h. The structure of each flow is then as follows:
     z <- autoregressive_layer(z) + h, allow for diagonal connections
     z <- autoregressive_layer(z), allow for diagonal connections
     :
     z <- autoregressive_layer(z), do not allow for diagonal connections.

     Note that the size of h needs to be the same as h_size, which is the width of the MADE layers.
     """

    def __init__(self, z_size, num_flows=2, num_hidden=0, h_size=50, forget_bias=1., conv2d=False):
        super(IAF, self).__init__()
        self.z_size = z_size
        self.num_flows = num_flows
        self.num_hidden = num_hidden
        self.h_size = h_size
        self.conv2d = conv2d
        if not conv2d:
            ar_layer = MaskedLinear
        else:
            ar_layer = MaskedConv2d
        self.activation = torch.nn.ELU
        # self.activation = torch.nn.ReLU

        self.forget_bias = forget_bias
        self.flows = []
        self.param_list = []

        # For reordering z after each flow
        flip_idx = torch.arange(self.z_size - 1, -1, -1).long()
        self.register_buffer('flip_idx', flip_idx)

        for k in range(num_flows):
            arch_z = [ar_layer(z_size, h_size), self.activation()]
            self.param_list += list(arch_z[0].parameters())
            z_feats = torch.nn.Sequential(*arch_z)
            arch_zh = []
            for j in range(num_hidden):
                arch_zh += [ar_layer(h_size, h_size), self.activation()]
                self.param_list += list(arch_zh[-2].parameters())
            zh_feats = torch.nn.Sequential(*arch_zh)
            linear_mean = ar_layer(h_size, z_size, diagonal_zeros=True)
            linear_std = ar_layer(h_size, z_size, diagonal_zeros=True)
            self.param_list += list(linear_mean.parameters())
            self.param_list += list(linear_std.parameters())

            if torch.cuda.is_available():
                z_feats = z_feats.cuda()
                zh_feats = zh_feats.cuda()
                linear_mean = linear_mean.cuda()
                linear_std = linear_std.cuda()
            self.flows.append((z_feats, zh_feats, linear_mean, linear_std))

        self.param_list = torch.nn.ParameterList(self.param_list)

    def forward(self, z, h_context):

        logdets = 0.
        for i, flow in enumerate(self.flows):
            if (i + 1) % 2 == 0 and not self.conv2d:
                # reverse ordering to help mixing
                z = z[:, self.flip_idx]

            h = flow[0](z)
            h = h + h_context
            h = flow[1](h)
            mean = flow[2](h)
            gate = F.sigmoid(flow[3](h) + self.forget_bias)
            z = gate * z + (1 - gate) * mean
            logdets += torch.sum(safe_log(gate).view(gate.size(0), -1), 1)
        return z, logdets


class LinIAF(nn.Module):
    def __init__(self, z_size):
        super(LinIAF, self).__init__()
        self.z_size = z_size

    def forward(self, z, L):
        '''
        :param L: batch_size (B) x latent_size^2 (L^2)
        :param z: batch_size (B) x latent_size (L)
        :return: z_new = L*z
        '''
        #bs = L.size(0)
        
        # L->tril(L)
        L_matrix = L.view( -1, self.z_size, self.z_size ) # resize to get B x L x L
        LTmask = torch.tril( torch.ones(self.z_size, self.z_size), diagonal=-1 ) # lower-triangular mask matrix (1s in lower triangular part)
        I = torch.eye(self.z_size, self.z_size).expand(L_matrix.size(0), self.z_size, self.z_size)

        if torch.cuda.is_available():
            LTmask = LTmask.cuda()
            I = I.cuda()

        LTmask = LTmask.unsqueeze(0).expand( L_matrix.size(0), self.z_size, self.z_size ) # 1 x L x L -> B x L x L
        LT = torch.mul( L_matrix, LTmask ) + I # here we get a batch of lower-triangular matrices with ones on diagonal

        # z_new = L * z
        z_new = torch.bmm( LT , z.unsqueeze(2) ).squeeze(2) # B x L x L * B x L x 1 -> B x L
        return z_new


class Affine(nn.Module):
    def __init__(self):
        super(Affine, self).__init__()

    def get_params(self, flow_coef):
        a = flow_coef[..., 0] # [B, D]
        log_b_sq = flow_coef[..., 1]
        b = torch.exp(0.5 * log_b_sq)
        return a, log_b_sq, b

    def forward(self, z, flow_coef):
        a, log_b_sq, b = self.get_params(flow_coef)
        z_new = a + b * z
        log_det_jacobian = 0.5 * log_b_sq.sum(-1)
        return z_new, log_det_jacobian

    def inverse(self, z, flow_coef):
        a, log_b_sq, b = self.get_params(flow_coef)
        z_prev = (z - a) / b
        log_det_jacobian = 0.5 * log_b_sq.sum(-1)
        return z_prev, log_det_jacobian


def arccosh(x):
    return torch.log(x + torch.sqrt(x.pow(2)-1))


def arcsinh(x):
    return torch.log(x + torch.sqrt(x.pow(2)+1))


class NLSq(nn.Module):
    def __init__(self):
        super(NLSq, self).__init__()

    def get_params(self, flow_coef):
        a = flow_coef[..., 0]
        log_b = flow_coef[..., 1] #* 0.4
        c_prime = flow_coef[..., 2] #* 0.3
        log_d = flow_coef[..., 3] #* 0.4
        g = flow_coef[..., 4]

        b = torch.exp(log_b)
        d = torch.exp(log_d)
        log_A = math.log(8 * math.sqrt(3) / 9 - 0.05)  # 0.05 is a small number to prevent exactly 0 slope
        c = torch.tanh(c_prime) * torch.exp(log_A + log_b - log_d)

        return a, b, c, d, g

    def inverse(self, z, flow_coef):
        """
        Technically, computing the reverse direction of the NLSq function defined in:
        https://arxiv.org/pdf/1901.10548.pdf
        """
        a, b, c, d, g = self.get_params(flow_coef)

        #if True:
        #    z_prev = (z - a) / b
        #    log_det_jacobian = torch.log(b).sum(-1)
        #    return z_prev, log_det_jacobian


        # double needed for stability. No effect on overall speed
        a = a.double()
        b = b.double()
        c = c.double()
        #c = torch.zeros_like(c).double()
        d = d.double()
        g = g.double()
        z = z.double()

        aa = -b * d.pow(2)
        bb = (z - a) * d.pow(2) - 2 * b * d * g
        cc = (z - a) * 2 * d * g - b * (1 + g.pow(2))
        dd = (z - a) * (1 + g.pow(2)) - c

        p = (3 * aa * cc - bb.pow(2)) / (3 * aa.pow(2))
        q = (2 * bb.pow(3) - (9 * aa * bb * cc) + (27 * aa.pow(2) * dd)) / (27 * aa.pow(3))
        
        t = -2 * torch.abs(q) / q * torch.sqrt(torch.abs(p) / 3)
        inter_term1 = -3 * torch.abs(q) / (2*p) * torch.sqrt(3 / torch.abs(p))
        inter_term2 = 1/3 * arccosh(torch.abs(inter_term1 - 1) + 1)
        t = t * torch.cosh(inter_term2)

        tpos = -2 * torch.sqrt(torch.abs(p) / 3)
        inter_term1 = 3*q / (2*p) * torch.sqrt(3 / torch.abs(p))
        inter_term2 = 1/3 * arcsinh(inter_term1)
        tpos = tpos * torch.sinh(inter_term2)

        t[p > 0] = tpos[p > 0]
        z_new = t - bb / (3*aa)

        arg = d * z_new + g
        denom = 1 + arg.pow(2)
        log_det_jacobian = torch.log(b - 2 * c * d * arg / denom.pow(2)).sum(-1)
        #z = a + b*z_new + c/denom

        z_new = torch.max(z_new, torch.ones_like(z_new) * -100.0)
        z_new = torch.min(z_new, torch.ones_like(z_new) * 100.0)


        return z_new.float(), log_det_jacobian.float()

    def forward(self, z, flow_coef):
        a, b, c, d, g = self.get_params(flow_coef)
        #c = torch.zeros_like(c)

        arg = d*z + g
        denom = 1 + arg.pow(2)
        z_new = a + b*z + c/denom

        z_new = torch.max(z_new, torch.ones_like(z_new) * -100.0)
        z_new = torch.min(z_new, torch.ones_like(z_new) * 100.0)
        
        log_det_jacobian = torch.log(b - 2 * c * d * arg / denom.pow(2)).sum(-1)
        return z_new, log_det_jacobian

        
