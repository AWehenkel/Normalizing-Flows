import torch
import torch.nn as nn
from ..Conditionners import AutoregressiveConditioner
from ..Normalizers import AffineNormalizer
from ..Step import FCNormalizingFlow, NormalizingFlowStep
from ..Utils.Distributions import NormalLogDensity


# Todo defined as an augmented flow step which can be composed with other flow or augmented flow into a simple flow.
# An augmented flow is a superclass of a flow (a flow is an augmented with no augmentation)
class StochasticUpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, out_size, flow):
        super(StochasticUpSampling, self).__init__()
        self.up_sampler = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                             stride, padding)
        self.up_sampler_cond = nn.Upsample(out_size)
        self.flow = flow

    def compute_ll(self, x, context):
        x_approx = self.up_sampler(context)
        up_sampled_context = self.up_sampler_cond(context)
        #x_approx += torch.randn(x_approx.shape).to(x.device) / 20.
        residual = x - x_approx
        ll, z = self.flow.compute_ll(residual, up_sampled_context)
        return ll, z

    def invert(self, z, context):
        x_approx = self.up_sampler(context)
        #x_approx += torch.randn(x_approx.shape).to(z.device) / 20.
        up_sampled_context = self.up_sampler_cond(context)
        residual = self.flow.invert(z, up_sampled_context)
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
        print(c, h, w)

        conditioner = AutoregressiveConditioner(C, [150, 150, 150], 2, c)
        normalizer = AffineNormalizer()
        flow_steps = [NormalizingFlowStep(conditioner, normalizer)]
        conditioner = AutoregressiveConditioner(C, [150, 150, 150], 2, c)
        normalizer = AffineNormalizer()
        flow_steps += [NormalizingFlowStep(conditioner, normalizer)]
        conditioner = AutoregressiveConditioner(C, [150, 150, 150, 150], 2, c)
        normalizer = AffineNormalizer()
        flow_steps += [NormalizingFlowStep(conditioner, normalizer)]
        flow = FCNormalizingFlow(flow_steps, NormalLogDensity())

        img_flow = ContextualIndependant2DFlowStep(flow)

        self.up_sampler = StochasticUpSampling(c, C, kernel_size, stride, pad_up, [H, W], img_flow)

    def compute_ll(self, x, context=None):
        context = self.down_sampler(x)
        #context += torch.randn(context.shape).to(x.device) / 10.
        ll, z = self.up_sampler.compute_ll(x, context)
        return ll, z, context

    def invert(self, z, context=None):
        return self.up_sampler.invert(z, context)


class MNISTAugmentedFlow(nn.Module):
    def __init__(self):
        super(MNISTAugmentedFlow, self).__init__()
        self.c1, self.c2, self.c3 = 5, 5, 5
        self.o1, self.o2, self.o3, self.o4 = 28, 26, 13, 36

        self.l1 = Augmented2DFlowStep([1, self.o1, self.o1], self.c1, 3, 1, 0)
        self.l2 = Augmented2DFlowStep([self.c1, self.o2, self.o2], self.c2, 2, 2, 0, nn.ReLU())
        self.l3 = Augmented2DFlowStep([self.c2,  self.o3,  self.o3], self.c3, 3, 2, 0, nn.ReLU())

        conditioner = AutoregressiveConditioner(self.o4*self.c3, [150, 150, 150, 150], 2)
        normalizer = AffineNormalizer()
        flow_steps = [NormalizingFlowStep(conditioner, normalizer)]
        conditioner = AutoregressiveConditioner(self.o4 * self.c3, [150, 150, 150, 150], 2)
        normalizer = AffineNormalizer()
        flow_steps += [NormalizingFlowStep(conditioner, normalizer)]
        conditioner = AutoregressiveConditioner(self.o4 * self.c3, [150, 150, 150, 150], 2)
        normalizer = AffineNormalizer()
        flow_steps += [NormalizingFlowStep(conditioner, normalizer)]
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
        z1 = z[:, :-self.o4*self.c3 - self.c2 * self.o3 * self.o3 - self.c1 * self.o2 * self.o2].reshape(b_size, 1, self.o1, self.o1)
        z2 = z[:, -self.o4*self.c3 - self.c2 * self.o3 * self.o3 - self.c1 * self.o2 * self.o2:-self.o4 * self.c3 - self.c2 * self.o3 *  self.o3].reshape(b_size, self.c1, self.o2, self.o2)
        z3 = z[:, -self.o4*self.c3 - self.c2 * self.o3 * self.o3:-self.o4*self.c3].reshape(b_size, self.c2, self.o3, self.o3)
        z4 = z[:, -self.o4*self.c3:]

        context3 = self.l4.invert(z4, context).reshape(b_size, self.c3, int(self.o4**.5), int(self.o4**.5))
        context2 = self.l3.invert(z3, context3)
        context1 = self.l2.invert(z2, context2)
        x = self.l1.invert(z1, context1)
        return x

    def forward(self, x, context=None):
        return self.compute_ll(x, context)



class MNISTBaseline(nn.Module):
    def __init__(self):
        super(MNISTBaseline, self).__init__()

        conditioner = AutoregressiveConditioner(784, [1000, 1000, 1000], 2)
        normalizer = AffineNormalizer()
        flow_steps = [NormalizingFlowStep(conditioner, normalizer)]
        conditioner = AutoregressiveConditioner(784, [1000, 1000, 1000], 2)
        normalizer = AffineNormalizer()
        flow_steps += [NormalizingFlowStep(conditioner, normalizer)]
        conditioner = AutoregressiveConditioner(784, [1000, 1000, 1000], 2)
        normalizer = AffineNormalizer()
        flow_steps += [NormalizingFlowStep(conditioner, normalizer)]
        self.l4 = FCNormalizingFlow(flow_steps, NormalLogDensity())

    def compute_ll(self, x, context=None):
        b_size = x.shape[0]
        ll, z = self.l4.compute_ll(x.reshape(b_size, -1))
        return ll, z.reshape(b_size, -1)

    def invert(self, z, context=None):
        b_size = z.shape[0]
        return self.l4.invert(z.reshape(b_size, -1)).reshape(b_size, 1, 28, 28)

    def forward(self, x, context=None):
        return self.compute_ll(x, context)
