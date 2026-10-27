"""Players: simple baseline bots and the neural-network policy.

Every agent has act(env, seat, eqc, rng) -> action, where eqc is the hand's EquityCache.
"""
import numpy as np

from .env import ALL_IN, CALL, FOLD, HALF_POT, POT
from .features import encode, legal_mask


class RandomAgent:
    def act(self, env, seat, eqc, rng):
        return rng.choice(env.legal_actions())


class CallAgent:
    """Never folds, never raises."""

    def act(self, env, seat, eqc, rng):
        return CALL


class RaiseAgent:
    """Raises the pot whenever it can."""

    def act(self, env, seat, eqc, rng):
        legal = env.legal_actions()
        return POT if POT in legal else ALL_IN if ALL_IN in legal else CALL


class EquityAgent:
    """Bets strong hands, calls when the price is right, folds the rest."""

    def act(self, env, seat, eqc, rng):
        legal = env.legal_actions()
        eq = eqc.get(env, seat)
        tc = env.to_call(seat)
        if eq > 0.8 and POT in legal:
            return POT
        if eq > 0.65 and HALF_POT in legal and len(env.history[env.street]) < 3:
            return HALF_POT
        if tc == 0 or eq > tc / (env.pot + tc) + 0.05:
            return CALL
        return FOLD


def forward(w, x, mask):
    """Numpy forward pass of model.PolicyNet. Returns (action probabilities, value)."""
    h = np.maximum(x @ w['body.0.weight'].T + w['body.0.bias'], 0)
    h = np.maximum(h @ w['body.2.weight'].T + w['body.2.bias'], 0)
    logits = h @ w['pi.weight'].T + w['pi.bias']
    logits = np.where(mask > 0, logits, -1e9)
    logits -= logits.max(axis=-1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=-1, keepdims=True)
    v = (h @ w['v.weight'].T + w['v.bias'])[..., 0]
    return p, v


class PolicyAgent:
    """Samples from the network's mixed strategy. With record=True it keeps training data."""

    def __init__(self, weights, greedy=False, record=False):
        self.w = weights
        self.greedy = greedy
        self.record = record
        self.trace = []

    def act(self, env, seat, eqc, rng):
        x = encode(env, seat, eqc.get(env, seat))
        m = legal_mask(env)
        p, v = forward(self.w, x, m)
        a = int(np.argmax(p)) if self.greedy else int(rng.choices(range(len(p)), weights=p)[0])
        if self.record:
            self.trace.append((x, m, a, float(np.log(p[a] + 1e-12)), float(v)))
        return a


BASELINES = {
    'random': RandomAgent,
    'call': CallAgent,
    'raise': RaiseAgent,
    'equity': EquityAgent,
}


def make_agent(spec):
    """spec is a baseline name, or ('policy', weights)."""
    if isinstance(spec, str):
        return BASELINES[spec]()
    return PolicyAgent(spec[1])
