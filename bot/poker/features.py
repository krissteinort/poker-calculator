"""Turn a game state into the fixed-size vector the policy network sees.

Kept deliberately simple so it can be re-implemented in JS for the browser.
"""
import random

import numpy as np

from .cards import category, equity_mc, preflop_equity
from .env import CALL, FOLD, N_ACTIONS

OBS_SIZE = 52 + 52 + 1 + 4 + 1 + 5 + 9 + 16
CHIP_SCALE_BB = 100  # chip amounts are fed to the network in units of 100 big blinds


class EquityCache:
    """Equity per (seat, street) for one hand: it only changes when new cards appear."""

    def __init__(self, samples=128, rng=random):
        self.samples = samples
        self.rng = rng
        self.cache = {}

    def get(self, env, seat):
        key = (seat, env.street)
        if key not in self.cache:
            board = env.board
            if not board:
                self.cache[key] = preflop_equity(env.hole[seat])
            else:
                self.cache[key] = equity_mc(env.hole[seat], board, self.samples, self.rng)
        return self.cache[key]


def encode(env, seat, eq):
    x = np.zeros(OBS_SIZE, dtype=np.float32)
    for c in env.hole[seat]:
        x[c] = 1
    board = env.board
    for c in board:
        x[52 + c] = 1
    i = 104
    x[i] = eq; i += 1
    x[i + env.street] = 1; i += 4
    x[i] = env.button == seat; i += 1

    s0 = CHIP_SCALE_BB * env.bb  # fixed scale, so deeper stacks show up as bigger values
    tc = env.to_call(seat)
    x[i] = env.pot / (2 * s0)
    x[i + 1] = env.stack[seat] / s0
    x[i + 2] = env.stack[1 - seat] / s0
    x[i + 3] = tc / s0
    x[i + 4] = tc / (env.pot + tc)  # pot odds
    i += 5

    cards = env.hole[seat] + board
    x[i + (category(cards) if len(cards) >= 5 else int(cards[0] >> 2 == cards[1] >> 2))] = 1
    i += 9

    # Per street: my raises, opponent raises, my calls, opponent calls (scaled).
    for st in range(4):
        for who, a in env.history[st]:
            if a == FOLD:
                continue
            j = (0 if who == seat else 1) + (2 if a == CALL else 0)
            x[i + st * 4 + j] += 0.25
    return x


def legal_mask(env):
    m = np.zeros(N_ACTIONS, dtype=np.float32)
    m[env.legal_actions()] = 1
    return m
