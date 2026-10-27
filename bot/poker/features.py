"""Turn a game state into the fixed-size vector the policy network sees.

Kept deliberately simple so it can be re-implemented in JS for the browser (play-engine.js).
Opponents are described in seat order clockwise from the player, padded to MAX_PLAYERS - 1 slots.
"""
import random

import numpy as np

from .cards import category, equity_mc, preflop_equity
from .env import CALL, FOLD, MAX_PLAYERS, N_ACTIONS

OPP_FEATURES = 7
OBS_SIZE = (52 + 52 + 1 + 4 + 5 + 5 + MAX_PLAYERS + 1 + 5 + 9 + 16
            + (MAX_PLAYERS - 1) * OPP_FEATURES)
CHIP_SCALE_BB = 100  # chip amounts are fed to the network in units of 100 big blinds


def opponents_in(env, seat):
    return sum(1 for s in env.in_hand() if s != seat)


class EquityCache:
    """Equity per (seat, street, opponents left) for one hand: it only changes when cards appear or players fold."""

    def __init__(self, samples=128, rng=random):
        self.samples = samples
        self.rng = rng
        self.cache = {}

    def get(self, env, seat):
        k = opponents_in(env, seat)
        key = (seat, env.street, k)
        if key not in self.cache:
            board = env.board
            if not board:
                self.cache[key] = preflop_equity(env.hole[seat], k)
            else:
                self.cache[key] = equity_mc(env.hole[seat], board, self.samples, self.rng, k)
        return self.cache[key]


def encode(env, seat, eq):
    x = np.zeros(OBS_SIZE, dtype=np.float32)
    n = env.n
    for c in env.hole[seat]:
        x[c] = 1
    board = env.board
    for c in board:
        x[52 + c] = 1
    i = 104
    x[i] = eq; i += 1
    x[i + env.street] = 1; i += 4
    x[i + n - 2] = 1; i += 5                          # players at the table
    x[i + len(env.in_hand()) - 2] = 1; i += 5         # players still in the hand
    x[i + (seat - env.button) % n] = 1; i += MAX_PLAYERS  # seats after the button (0 = button)
    x[i] = sum(1 for o in range(n) if o != seat and env.can_act(o) and not env.acted[o]) / 5; i += 1

    s0 = CHIP_SCALE_BB * env.bb  # fixed scale, so deeper stacks show up as bigger values
    tc = env.to_call(seat)
    x[i] = env.stack[seat] / s0
    x[i + 1] = env.pot / (2 * s0)
    x[i + 2] = tc / s0
    x[i + 3] = tc / (env.pot + tc)  # pot odds
    x[i + 4] = env.bet[seat] / s0
    i += 5

    cards = env.hole[seat] + board
    x[i + (category(cards) if len(cards) >= 5 else int(cards[0] >> 2 == cards[1] >> 2))] = 1
    i += 9

    # Per street: my raises, my calls, everyone else's raises, everyone else's calls (scaled).
    for st in range(4):
        for who, a in env.history[st]:
            if a == FOLD:
                continue
            x[i + st * 4 + (0 if who == seat else 2) + (1 if a == CALL else 0)] += 0.25
    i += 16

    for k in range(1, MAX_PLAYERS):
        if k < n:
            o = (seat + k) % n
            x[i] = 1
            x[i + 1] = not env.folded[o]
            x[i + 2] = not env.folded[o] and env.stack[o] == 0
            x[i + 3] = env.stack[o] / s0
            x[i + 4] = env.bet[o] / s0
            x[i + 5] = env.contrib[o] / s0
            x[i + 6] = 0.25 * sum(1 for who, a in env.history[env.street] if who == o and a > CALL)
        i += OPP_FEATURES
    return x


def legal_mask(env):
    m = np.zeros(N_ACTIONS, dtype=np.float32)
    m[env.legal_actions()] = 1
    return m
