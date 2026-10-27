"""Playing hands: training rollouts and duplicate matches. The *_chunk/rollout functions run in worker processes."""
import math
import random

import numpy as np

from .agents import PolicyAgent, make_agent
from .env import HeadsUpEnv
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
    """The learner plays n_hands against one opponent. Returns its decisions as training arrays.
    Each seat's stack is drawn log-uniformly from stack_range (in big blinds)."""
    weights, opp_spec, n_hands, seed, stack_range = args
    lo, hi = math.log(stack_range[0]), math.log(stack_range[1])
    rng = random.Random(seed)
    env = HeadsUpEnv()
    learner = PolicyAgent(weights, record=True)
    opp = make_agent(opp_spec)
    rows, rews, dones = [], [], []
    total = 0.0
    for i in range(n_hands):
        seat, button = (i // 2) % 2, i % 2
        agents = [opp, opp]
        agents[seat] = learner
        learner.trace = []
        stacks = [round(math.exp(rng.uniform(lo, hi))) * env.bb for _ in range(2)]
        payoff = play_hand(env, agents, None, button, rng, stacks)
        total += payoff[seat]
        n = len(learner.trace)
        for k, row in enumerate(learner.trace):
            rows.append(row)
            rews.append(payoff[seat] / REWARD_SCALE if k == n - 1 else 0.0)
            dones.append(k == n - 1)
    x, m, a, lp, v = zip(*rows) if rows else ([], [], [], [], [])
    return {
        'obs': np.array(x, dtype=np.float32), 'mask': np.array(m, dtype=np.float32),
        'act': np.array(a, dtype=np.int64), 'logp': np.array(lp, dtype=np.float32),
        'val': np.array(v, dtype=np.float32), 'rew': np.array(rews, dtype=np.float32),
        'done': np.array(dones, dtype=bool), 'winnings': total, 'hands': n_hands,
    }


def duplicate_chunk(args):
    """Agent A vs B on n_pairs decks, each deck played twice with the seats swapped.
    Returns A's average winnings per hand (big blinds) for each pair."""
    spec_a, spec_b, n_pairs, seed, stacks = args
    rng = random.Random(seed)
    env = HeadsUpEnv()
    a, b = make_agent(spec_a), make_agent(spec_b)
    out = []
    for i in range(n_pairs):
        deck = list(range(52))
        rng.shuffle(deck)
        button = i % 2
        r = play_hand(env, [a, b], deck, button, rng, stacks)[0]
        r += play_hand(env, [b, a], deck, button, rng, stacks)[1]
        out.append(r / 2)
    return out


def duplicate_match(pool, spec_a, spec_b, n_pairs, seed, chunks, stack_bb=100):
    per = max(1, n_pairs // chunks)
    stacks = [round(stack_bb) * HeadsUpEnv().bb] * 2
    tasks = [(spec_a, spec_b, per, seed * 7919 + i, stacks) for i in range(chunks)]
    res = np.concatenate([np.array(r) for r in pool.map(duplicate_chunk, tasks)])
    mbb = res.mean() * 1000
    ci = 1.96 * res.std(ddof=1) / np.sqrt(len(res)) * 1000
    return mbb, ci
