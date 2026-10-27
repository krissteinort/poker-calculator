"""No-limit hold'em for 2-6 players, one hand at a time.

Seats 0..n-1 go clockwise. Heads-up, the button posts the small blind and acts first preflop; with 3+ players
the two seats after the button post the blinds and the seat after the big blind acts first. After the flop the
first player still in after the button acts first. Unequal all-ins are settled with side pots.
Bet sizes are abstracted to five actions. Simplification: a short all-in still reopens the betting.
"""
import random

from .cards import evaluate

FOLD, CALL, HALF_POT, POT, ALL_IN = range(5)
ACTION_NAMES = ['fold', 'check/call', 'raise ½ pot', 'raise pot', 'all-in']
N_ACTIONS = 5
MAX_PLAYERS = 6
BOARD_COUNT = [0, 3, 4, 5, 5]  # cards visible on each street (4 = showdown)


class PokerEnv:
    def __init__(self, n_players=2, stack_bb=100, bb=2):
        self.n = n_players
        self.bb = bb
        self.start_stack = stack_bb * bb

    def reset(self, deck=None, button=0, rng=random, stacks=None):
        """stacks: starting chips per seat; its length sets the number of players. Each must cover a big blind."""
        if stacks:
            self.n = len(stacks)
        n = self.n
        assert 2 <= n <= MAX_PLAYERS
        if deck is None:
            deck = list(range(52))
            rng.shuffle(deck)
        self.hole = [deck[2 * i:2 * i + 2] for i in range(n)]
        self.full_board = deck[2 * n:2 * n + 5]
        self.button = button
        self.stack = list(stacks) if stacks else [self.start_stack] * n
        assert min(self.stack) >= self.bb, self.stack
        self.bet = [0] * n       # chips put in on this street
        self.contrib = [0] * n   # chips put in this hand
        self.folded = [False] * n
        self.acted = [False] * n
        self.street = 0
        self.history = [[] for _ in range(4)]  # (seat, action) per street
        self.done = False
        self.payoff = None       # net result per seat, in big blinds
        self.sb_seat = button if n == 2 else (button + 1) % n
        self.bb_seat = (self.sb_seat + 1) % n
        self._put(self.sb_seat, self.bb // 2)
        self._put(self.bb_seat, self.bb)
        self.last_raise = self.bb
        self.to_act = self._next_needing(self.bb_seat)

    @property
    def board(self):
        return self.full_board[:BOARD_COUNT[self.street]]

    @property
    def pot(self):
        return sum(self.contrib)

    def to_call(self, seat):
        return max(self.bet) - self.bet[seat]

    def can_act(self, seat):
        return not self.folded[seat] and self.stack[seat] > 0

    def in_hand(self):
        return [s for s in range(self.n) if not self.folded[s]]

    def _put(self, seat, amount):
        amount = min(amount, self.stack[seat])
        self.stack[seat] -= amount
        self.bet[seat] += amount
        self.contrib[seat] += amount

    def _next_needing(self, after):
        """First seat clockwise after `after` that still has to act this round."""
        mb = max(self.bet)
        for i in range(1, self.n + 1):
            s = (after + i) % self.n
            if self.can_act(s) and (not self.acted[s] or self.bet[s] < mb):
                return s
        return None

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
        others_can_act = any(self.can_act(o) for o in range(self.n) if o != seat)
        if self.stack[seat] > tc and others_can_act:
            if self.raise_amount(seat, 0.5) < self.stack[seat]:
                acts.append(HALF_POT)
            if self.raise_amount(seat, 1.0) < self.stack[seat]:
                acts.append(POT)
            acts.append(ALL_IN)
        return acts

    def step(self, a):
        seat = self.to_act
        assert not self.done and a in self.legal_actions(), (a, self.legal_actions())
        self.history[self.street].append((seat, a))
        if a == FOLD:
            self.folded[seat] = True
            left = self.in_hand()
            if len(left) == 1:
                win = [0] * self.n
                win[left[0]] = self.pot
                return self._finish(win)
        elif a == CALL:
            self._put(seat, self.to_call(seat))
            self.acted[seat] = True
        else:
            amount = self.stack[seat] if a == ALL_IN else self.raise_amount(seat, 0.5 if a == HALF_POT else 1.0)
            raise_by = self.bet[seat] + amount - max(self.bet)
            self._put(seat, amount)
            self.last_raise = max(self.last_raise, raise_by)
            self.acted = [False] * self.n
            self.acted[seat] = True

        nxt = self._next_needing(seat)
        if nxt is None:
            self._end_round()
        else:
            self.to_act = nxt

    def _end_round(self):
        if sum(self.can_act(s) for s in range(self.n)) <= 1 or self.street == 3:
            self.street = 4
            return self._showdown()
        self.street += 1
        self.bet = [0] * self.n
        self.acted = [False] * self.n
        self.last_raise = self.bb
        self.to_act = self._next_needing(self.button)

    def _showdown(self):
        """Split the main pot and any side pots between the best hands eligible for each."""
        live = self.in_hand()
        score = {s: evaluate(self.hole[s] + self.full_board) for s in live}
        win = [0] * self.n
        prev = 0
        for level in sorted(set(self.contrib[s] for s in live)):
            part = sum(min(c, level) - min(c, prev) for c in self.contrib)
            eligible = [s for s in live if self.contrib[s] >= level]
            best = max(score[s] for s in eligible)
            winners = [s for s in eligible if score[s] == best]
            share, odd = divmod(part, len(winners))
            # odd chips go to the winners closest to the left of the button
            winners.sort(key=lambda s: (s - self.button - 1) % self.n)
            for k, s in enumerate(winners):
                win[s] += share + (1 if k < odd else 0)
            prev = level
        self._finish(win)

    def _finish(self, win):
        """win: chips each seat takes from the pot."""
        self.done = True
        self.payoff = [(win[s] - self.contrib[s]) / self.bb for s in range(self.n)]
