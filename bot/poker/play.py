"""Playing hands: training rollouts and duplicate matches. The *_chunk/rollout functions run in worker processes."""
import math
import random

import numpy as np

from .agents import PolicyAgent, make_agent
from .env import PokerEnv
from .features import EquityCache

REWARD_SCALE = 100.0  # payoffs are in big blinds; a 100bb stack -> rewards in [-1, 1]


def play_hand(env, agents, deck, button, rng, stacks=None):
    env.reset(deck, button, rng, stacks)
    eqc = EquityCache(rng=rng)
    while not env.done:
        s = env.to_act
        env.step(agents[s].act(env, s, eqc, rng))
    return env.payoff


def rollout(args):
    """The learner plays n_hands. Each hand has a random table size and stacks; the learner takes one seat and
    each other seat is either the learner again (self_frac) or the task's opponent. Every learner seat is recorded.
    Stacks are drawn log-uniformly from stack_range (big blinds)."""
    weights, opp_spec, n_hands, seed, stack_range, players_range, self_frac = args
    lo, hi = math.log(stack_range[0]), math.log(stack_range[1])
    rng = random.Random(seed)
    env = PokerEnv()
    learner = PolicyAgent(weights, record=True)
    opp = make_agent(opp_spec)
    rows, rews, dones = [], [], []
    total, seats_played = 0.0, 0
    for _ in range(n_hands):
        n = rng.randint(*players_range)
        me = rng.randrange(n)
        agents = [learner if s == me or rng.random() < self_frac else opp for s in range(n)]
        learner.trace = []
        stacks = [round(math.exp(rng.uniform(lo, hi))) * env.bb for _ in range(n)]
        payoff = play_hand(env, agents, None, rng.randrange(n), rng, stacks)
        for s in range(n):
            if agents[s] is learner:
                total += payoff[s]
                seats_played += 1
                trace = [row for seat, row in learner.trace if seat == s]
                for k, row in enumerate(trace):
                    rows.append(row)
                    last = k == len(trace) - 1
                    rews.append(payoff[s] / REWARD_SCALE if last else 0.0)
                    dones.append(last)
    x, m, a, lp, v = zip(*rows) if rows else ([], [], [], [], [])
    return {
        'obs': np.array(x, dtype=np.float32), 'mask': np.array(m, dtype=np.float32),
        'act': np.array(a, dtype=np.int64), 'logp': np.array(lp, dtype=np.float32),
        'val': np.array(v, dtype=np.float32), 'rew': np.array(rews, dtype=np.float32),
        'done': np.array(dones, dtype=bool), 'winnings': total, 'seats': seats_played, 'hands': n_hands,
    }


def duplicate_chunk(args):
    """Agent A against n-1 copies of B. Each deck is played n times, with A moved to every seat in turn.
    Returns A's average winnings per hand (big blinds) for each deck."""
    spec_a, spec_b, n_decks, seed, stacks = args
    rng = random.Random(seed)
    env = PokerEnv()
    a, b = make_agent(spec_a), make_agent(spec_b)
    n = len(stacks)
    out = []
    for i in range(n_decks):
        deck = list(range(52))
        rng.shuffle(deck)
        button = i % n
        r = 0.0
        for seat in range(n):
            agents = [b] * n
            agents[seat] = a
            r += play_hand(env, agents, deck, button, rng, stacks)[seat]
        out.append(r / n)
    return out


def duplicate_match(pool, spec_a, spec_b, n_decks, seed, chunks, stack_bb=100, players=2):
    per = max(1, n_decks // chunks)
    stacks = [round(stack_bb) * PokerEnv().bb] * players
    tasks = [(spec_a, spec_b, per, seed * 7919 + i, stacks) for i in range(chunks)]
    res = np.concatenate([np.array(r) for r in pool.map(duplicate_chunk, tasks)])
    mbb = res.mean() * 1000
    ci = 1.96 * res.std(ddof=1) / np.sqrt(len(res)) * 1000
    return mbb, ci
