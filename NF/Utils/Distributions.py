import torch.distributions as D
import torch
from math import pi
import torch.nn as nn


class FlowDensity(nn.Module):
    def __init__(self):
        super(FlowDensity, self).__init__()

    def forward(self, z):
        pass

    def sample(self, shape):
        pass


class NormalLogDensity(nn.Module):
    """
    JIT-compatible Normal log density (replaces torch.distributions version).

    This is now the default implementation providing:
    - Full JIT compilation support (torch.jit.script)
    - Same interface and behavior as original
    - Manual implementation without torch.distributions dependency
    """

    def __init__(self, loc=0.0, scale=1.0):
        super().__init__()
        self.register_buffer('loc', torch.tensor(loc))
        self.register_buffer('scale', torch.tensor(scale))
        self.register_buffer('log_scale', torch.tensor(scale).log())
        self.register_buffer('log_2pi', torch.tensor(2.0 * pi).log())

    def forward(self, z):
        """Compute log probability density manually for JIT compatibility"""
        # Manual implementation: -0.5 * log(2*pi) - log(scale) - 0.5 * ((z - loc) / scale)^2
        normalized = (z - self.loc) / self.scale
        log_prob = -0.5 * self.log_2pi - self.log_scale - 0.5 * normalized.pow(2)
        return log_prob.sum(1)  # Sum over features, return [batch_size]

    def sample(self, shape):
        """Generate samples from standard normal"""
        return torch.randn(shape, device=self.loc.device) * self.scale + self.loc


# Legacy: torch.distributions version (kept for compatibility, but not JIT compatible)
class LegacyNormalLogDensity(nn.Module):
    """Original torch.distributions version - use only if JIT not needed"""
    def __init__(self):
        super().__init__()
        self.register_buffer("pi", torch.tensor(pi))

    def forward(self, z):
        return torch.distributions.Normal(loc=0., scale=1.).log_prob(z).sum(1)

    def sample(self, shape):
        return torch.randn(shape)


class MixtureLogDensity(nn.Module):
    def __init__(self, n_mode=10):
        super(MixtureLogDensity, self).__init__()
        self.register_buffer("pi", torch.tensor(pi))
        self.register_buffer("mu", torch.arange(-3., 3.0001, 6. / float(n_mode - 1)))
        self.register_buffer("sigma", torch.ones(n_mode, ) * 1.5 / float(n_mode))
        self.register_buffer("mix_weights", torch.ones(n_mode, ))

    def forward(self, z):
        mix = D.Categorical(self.mix_weights)
        comp = D.Normal(self.mu, self.sigma)
        dist = D.MixtureSameFamily(mix, comp)
        return dist.log_prob(z).sum(1)

