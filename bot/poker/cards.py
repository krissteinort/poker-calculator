"""Card encoding, 7-card hand evaluator and equity estimates.

Card = rank * 4 + suit, rank 0 (2) .. 12 (A), suit 0..3 (s h d c).
Scores match the JS engine in index.html: category is score >> 20, higher is better.
"""
import json
import os
import random

RANKS = '23456789TJQKA'
SUITS = 'shdc'
CATEGORIES = ['High Card', 'Pair', 'Two Pair', 'Three of a Kind', 'Straight',
              'Flush', 'Full House', 'Four of a Kind', 'Straight Flush']


def card(s):
    return RANKS.index(s[0].upper()) * 4 + SUITS.index(s[1].lower())


def cards(s):
    return [card(x) for x in s.split()]


def card_str(c):
    return RANKS[c >> 2] + SUITS[c & 3]


# ---- lookup tables over 13-bit rank masks ----

def _straight_high(mask):
    m = (mask << 1) | ((mask >> 12) & 1)  # bit 0 = ace played low
    for h in range(13, 3, -1):
        if (m >> (h - 4)) & 31 == 31:
            return h - 1
    return -1


BITS = [bin(m).count('1') for m in range(8192)]
STRAIGHT = [_straight_high(m) for m in range(8192)]
RANKLIST = [tuple(r for r in range(12, -1, -1) if (m >> r) & 1) for m in range(8192)]


def _top5(m):
    v = 0
    for r in RANKLIST[m][:5]:
        v = (v << 4) | r
    return v << (4 * (5 - min(5, BITS[m])))


TOP5 = [_top5(m) for m in range(8192)]


def evaluate(cs):
    """Score of the best 5-card hand from 5-7 cards."""
    rc = [0] * 13
    sms = [0, 0, 0, 0]
    mask = 0
    for c in cs:
        r = c >> 2
        rc[r] += 1
        sms[c & 3] |= 1 << r
        mask |= 1 << r
    flush = -1
    for sm in sms:
        if BITS[sm] >= 5:
            sh = STRAIGHT[sm]
            if sh >= 0:
                return (8 << 20) | (sh << 16)
            flush = (5 << 20) | TOP5[sm]
            break
    if BITS[mask] == len(cs):  # no paired ranks
        if flush >= 0:
            return flush
        sh = STRAIGHT[mask]
        if sh >= 0:
            return (4 << 20) | (sh << 16)
        return TOP5[mask]
    quad = t1 = t2 = p1 = p2 = -1
    for r in range(12, -1, -1):
        n = rc[r]
        if n < 2:
            continue
        if n == 4:
            quad = r
        elif n == 3:
            if t1 < 0:
                t1 = r
            elif t2 < 0:
                t2 = r
        elif p1 < 0:
            p1 = r
        elif p2 < 0:
            p2 = r
    if quad >= 0:
        return (7 << 20) | (quad << 16) | (RANKLIST[mask & ~(1 << quad)][0] << 12)
    if t1 >= 0 and (t2 >= 0 or p1 >= 0):
        return (6 << 20) | (t1 << 16) | (max(t2, p1) << 12)
    if flush >= 0:
        return flush
    sh = STRAIGHT[mask]
    if sh >= 0:
        return (4 << 20) | (sh << 16)
    if t1 >= 0:
        k = RANKLIST[mask & ~(1 << t1)]
        return (3 << 20) | (t1 << 16) | (k[0] << 12) | (k[1] << 8)
    if p2 >= 0:
        k = RANKLIST[mask & ~(1 << p1) & ~(1 << p2)]
        return (2 << 20) | (p1 << 16) | (p2 << 12) | (k[0] << 8)
    k = RANKLIST[mask & ~(1 << p1)]
    return (1 << 20) | (p1 << 16) | (k[0] << 12) | (k[1] << 8) | (k[2] << 4)


def category(cs):
    return evaluate(cs) >> 20


# ---- equity vs random hands ----

def equity_mc(hole, board, samples, rng=random, opponents=1):
    """Monte Carlo equity (share of the pot won, ties split) of `hole` vs `opponents` random hands."""
    known = set(hole) | set(board)
    deck = [c for c in range(52) if c not in known]
    need = 5 - len(board)
    k2 = 2 * opponents
    total = 0.0
    for _ in range(samples):
        draw = rng.sample(deck, need + k2)
        b = board + draw[k2:]
        me = evaluate(hole + b)
        ties = 0
        for o in range(0, k2, 2):
            s = evaluate(draw[o:o + 2] + b)
            if s > me:
                break
            ties += s == me
        else:
            total += 1.0 / (ties + 1)
    return total / samples


def preflop_class(hole):
    """'AKs', 'T9o', '77' style name for a starting hand."""
    a, b = sorted(hole, key=lambda c: -(c >> 2))
    ra, rb = RANKS[a >> 2], RANKS[b >> 2]
    if ra == rb:
        return ra + rb
    return ra + rb + ('s' if (a & 3) == (b & 3) else 'o')


PREFLOP_FILE = os.path.join(os.path.dirname(__file__), 'preflop_equity.json')
_preflop = None


def preflop_equity(hole, opponents=1):
    global _preflop
    if _preflop is None:
        with open(PREFLOP_FILE) as f:
            _preflop = json.load(f)
    return _preflop[preflop_class(hole)][opponents - 1]


def _preflop_row(args):
    hole, samples, seed = args
    rng = random.Random(seed)
    return preflop_class(hole), [round(equity_mc(hole, [], samples, rng, k), 4) for k in range(1, 6)]


def build_preflop_table(samples=30000):
    """{class: [equity vs 1 random hand, vs 2, ... vs 5]}"""
    from multiprocessing import Pool
    hands = []
    for i in range(13):
        for j in range(i, 13):
            ri, rj = 12 - i, 12 - j
            if ri == rj:
                hands.append([ri * 4, rj * 4 + 1])
            else:
                hands += [[ri * 4, rj * 4], [ri * 4, rj * 4 + 1]]  # suited, offsuit
    with Pool() as pool:
        return dict(pool.map(_preflop_row, [(h, samples, n) for n, h in enumerate(hands)]))


if __name__ == '__main__':
    t = build_preflop_table()
    with open(PREFLOP_FILE, 'w') as f:
        json.dump(t, f, indent=0, sort_keys=True)
    print('wrote', len(t), 'hands to', PREFLOP_FILE)
