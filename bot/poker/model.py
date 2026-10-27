import torch
from torch import nn

from .env import N_ACTIONS
from .features import OBS_SIZE


class PolicyNet(nn.Module):
    """Small actor-critic MLP. agents.forward mirrors this in numpy, so keep the layer names in sync."""

    def __init__(self, hidden=256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(OBS_SIZE, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.pi = nn.Linear(hidden, N_ACTIONS)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x, mask):
        h = self.body(x)
        logits = self.pi(h).masked_fill(mask == 0, -1e9)
        return logits, self.v(h).squeeze(-1)


def numpy_weights(model):
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
