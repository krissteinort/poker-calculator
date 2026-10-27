"""Heads-up no-limit hold'em, one hand at a time.

Seats 0 and 1. The button posts the small blind, acts first preflop and last after the flop.
Bet sizes are abstracted to five actions. Simplification: a short all-in still reopens the betting.
"""
import random

from .cards import evaluate

FOLD, CALL, HALF_POT, POT, ALL_IN = range(5)
ACTION_NAMES = ['fold', 'check/call', 'raise ½ pot', 'raise pot', 'all-in']
N_ACTIONS = 5
BOARD_COUNT = [0, 3, 4, 5, 5]  # cards visible on each street (4 = showdown)


class HeadsUpEnv:
    def __init__(self, stack_bb=100, bb=2):
        self.bb = bb
        self.start_stack = stack_bb * bb

    def reset(self, deck=None, button=0, rng=random):
        if deck is None:
            deck = list(range(52))
            rng.shuffle(deck)
        self.hole = [deck[0:2], deck[2:4]]
        self.full_board = deck[4:9]
        self.button = button
        self.stack = [self.start_stack, self.start_stack]
        self.bet = [0, 0]       # chips put in on this street
        self.contrib = [0, 0]   # chips put in this hand
        self.street = 0
        self.history = [[] for _ in range(4)]  # (seat, action) per street
        self.done = False
        self.payoff = None      # net result per seat, in big blinds
        self._put(button, self.bb // 2)
        self._put(1 - button, self.bb)
        self.last_raise = self.bb
        self.acted = [False, False]
        self.to_act = button

    @property
    def board(self):
        return self.full_board[:BOARD_COUNT[self.street]]

    @property
    def pot(self):
        return self.contrib[0] + self.contrib[1]

    def to_call(self, seat):
        return self.bet[1 - seat] - self.bet[seat]

    def _put(self, seat, amount):
        amount = min(amount, self.stack[seat])
        self.stack[seat] -= amount
        self.bet[seat] += amount
        self.contrib[seat] += amount

    def raise_amount(self, seat, frac):
        """Chips `seat` must add to raise by `frac` of the pot (after calling), at least a min-raise."""
        tc = self.to_call(seat)
        raise_by = max(int(frac * (self.pot + tc)), self.last_raise, self.bb)
        return tc + raise_by

    def legal_actions(self):
        seat = self.to_act
        tc = self.to_call(seat)
        acts = [FOLD] if tc > 0 else []
        acts.append(CALL)
        if self.stack[seat] > tc and self.stack[1 - seat] > 0:
            if self.raise_amount(seat, 0.5) < self.stack[seat]:
                acts.append(HALF_POT)
            if self.raise_amount(seat, 1.0) < self.stack[seat]:
                acts.append(POT)
            acts.append(ALL_IN)
        return acts

    def step(self, a):
        seat, opp = self.to_act, 1 - self.to_act
        assert not self.done and a in self.legal_actions(), (a, self.legal_actions())
        self.history[self.street].append((seat, a))
        if a == FOLD:
            lost = self.contrib[seat]
            self._finish({seat: -lost, opp: lost})
            return
        if a == CALL:
            self._put(seat, self.to_call(seat))
            self.acted[seat] = True
        else:
            amount = self.stack[seat] if a == ALL_IN else self.raise_amount(seat, 0.5 if a == HALF_POT else 1.0)
            raise_by = self.bet[seat] + amount - self.bet[opp]
            self._put(seat, amount)
            self.last_raise = max(self.last_raise, raise_by)
            self.acted = [False, False]
            self.acted[seat] = True

        matched = self.bet[0] == self.bet[1] or min(self.stack) == 0 and self.bet[seat] <= self.bet[opp]
        if self.acted[0] and self.acted[1] and matched:
            self._next_street()
        else:
            self.to_act = opp

    def _next_street(self):
        if min(self.stack) == 0 or self.street == 3:
            self.street = 4
            self._showdown()
            return
        self.street += 1
        self.bet = [0, 0]
        self.acted = [False, False]
        self.last_raise = self.bb
        self.to_act = 1 - self.button

    def _showdown(self):
        m = min(self.contrib)  # any unmatched excess is returned
        s0 = evaluate(self.hole[0] + self.full_board)
        s1 = evaluate(self.hole[1] + self.full_board)
        if s0 == s1:
            self._finish({0: 0, 1: 0})
        else:
            w = 0 if s0 > s1 else 1
            self._finish({w: m, 1 - w: -m})

    def _finish(self, net):
        self.done = True
        self.payoff = [net[0] / self.bb, net[1] / self.bb]
