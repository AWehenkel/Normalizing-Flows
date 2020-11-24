import torch
import torch.nn as nn
from models.Conditionners import AutoregressiveConditioner
from models.Normalizers import AffineNormalizer
from models.Step import FCNormalizingFlow, NormalizingFlowStep
from models.Utils.Distributions import NormalLogDensity


# Todo defined as an augmented flow step which can be composed with other flow or augmented flow into a simple flow.
# An augmented flow is a superclass of a flow (a flow is an augmented with no augmentation)
class StochasticUpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, flow):
        super(StochasticUpSampling, self).__init__()
        self.up_sampler = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                             stride, padding)
        self.flow = flow

    def compute_ll(self, x, context):
        x_approx = self.up_sampler(context)
        residual = x - x_approx
        ll, z = self.flow.compute_ll(residual, x_approx)
        return ll, z

    def invert(self, z, context):
        x_approx = self.up_sampler(context)
        residual = self.flow.invert(z, x_approx)
        x = residual + x_approx
        return x


class ContextualIndependant2DFlowStep(nn.Module):
    def __init__(self, flow):
        super(ContextualIndependant2DFlowStep, self).__init__()
        self.flow = flow

    def compute_ll(self, x, context):
        b_size, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, C)
        context = context.permute(0, 2, 3, 1).reshape(-1, context.shape[1])
        ll, z = self.flow.compute_ll(x, context)
        z = z.reshape(b_size, H, W, C).permute(0, 3, 1, 2)
        ll = ll.reshape(b_size, -1).sum()
        return ll, z

    def invert(self, z, context):
        b_size, C, H, W = z.shape
        z = z.permute(0, 2, 3, 1).reshape(-1, C)
        context = context.permute(0, 2, 3, 1).reshape(-1, context.shape[1])
        x = self.flow.invert(z, context)
        x = x.reshape(b_size, H, W, C).permute(0, 3, 1, 2)
        return x


# For now it is a simple Autoregressive Affine Flow (check others for later)
class Augmented2DFlowStep(nn.Module):
    def __init__(self, dim_in, n_channels, kernel_size, stride, padding, add_transfo=None, pad_up=0):
        super(Augmented2DFlowStep, self).__init__()
        C, H, W = dim_in
        conv = nn.Conv2d(dim_in[0], n_channels, kernel_size, stride, padding)
        down_sampler = [conv] + [add_transfo] if add_transfo is not None else [conv]
        self.down_sampler = nn.Sequential(*down_sampler)

        _, c, h, w = self.down_sampler(torch.zeros(1, C, H, W)).shape

        conditioner = AutoregressiveConditioner(C, [50, 50, 50], 2, C)
        normalizer = AffineNormalizer()
        flow_steps = [NormalizingFlowStep(conditioner, normalizer)]
        flow = FCNormalizingFlow(flow_steps, NormalLogDensity())

        img_flow = ContextualIndependant2DFlowStep(flow)

        self.up_sampler = StochasticUpSampling(c, C, kernel_size, stride, pad_up, img_flow)

    def compute_ll(self, x, context=None):
        context = self.down_sampler(x)
        ll, z = self.up_sampler.compute_ll(x, context)
        return ll, z, context

    def invert(self, z, context=None):
        return self.up_sampler.invert(z, context)


class MNISTAugmentedFlow(nn.Module):
    def __init__(self):
        super(MNISTAugmentedFlow, self).__init__()
        self.l1 = Augmented2DFlowStep([1, 28, 28], 5, 3, 1, 0)
        self.l2 = Augmented2DFlowStep([5, 26, 26], 5, 2, 2, 0, nn.ReLU())
        self.l3 = Augmented2DFlowStep([5, 13, 13], 1, 3, 2, 0, nn.ReLU())

        conditioner = AutoregressiveConditioner(36, [150, 150, 150], 2)
        normalizer = AffineNormalizer()
        flow_steps = [NormalizingFlowStep(conditioner, normalizer)]
        self.l4 = FCNormalizingFlow(flow_steps, NormalLogDensity())

    def compute_ll(self, x, context=None):
        b_size = x.shape[0]
        ll1, z1, context1 = self.l1.compute_ll(x, context)
        ll2, z2, context2 = self.l2.compute_ll(context1, context)
        ll3, z3, context3 = self.l3.compute_ll(context2, context)
        ll4, z4 = self.l4.compute_ll(context3.reshape(b_size, -1))

        ll = ll1 + ll2 + ll3 + ll4
        z = torch.cat((z1.reshape(b_size, -1), z2.reshape(b_size, -1), z3.reshape(b_size, -1), z4), 1)
        return ll, z

    def invert(self, z, context=None):
        b_size = z.shape[0]
        z1 = z[:, :-36 - 5 * 13 * 13 - 5 * 26 * 26].reshape(b_size, 1, 28, 28)
        z2 = z[:, -36 - 5 * 13 * 13 - 5 * 26 * 26:-36 - 5 * 13 * 13].reshape(b_size, 5, 26, 26)
        z3 = z[:, -36 - 5 * 13 * 13:-36].reshape(b_size, 5, 13, 13)
        z4 = z[:, -36:]

        context3 = self.l4.invert(z4, context).reshape(b_size, 1, 6, 6)
        context2 = self.l3.invert(z3, context3)
        context1 = self.l2.invert(z2, context2)
        x = self.l1.invert(z1, context1)
        return x

    def forward(self, x, context=None):
        return self.compute_ll(x, context)
