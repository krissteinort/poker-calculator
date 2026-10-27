"""Checks that play-engine.js (the browser bot) matches the Python it was trained with."""
import json
import os
import random
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from poker.agents import RandomAgent, forward
from poker.cards import evaluate
from poker.env import HeadsUpEnv
from poker.features import OBS_SIZE, encode, legal_mask

ENGINE = os.path.join(os.path.dirname(__file__), '..', '..', 'play-engine.js')

JS_CHECK = r"""
const E = require(process.argv[1]);
const d = JSON.parse(require('fs').readFileSync(process.argv[2]));
const bad = [];
d.evals.forEach(([cs, s]) => { if (E.evaluate(cs) !== s) bad.push(['eval', cs]); });
const policy = new E.Policy(d.weights);
d.hands.forEach((h, n) => {
  const env = new E.Env();
  env.reset(h.deck, h.button, h.stacks);
  h.decisions.forEach((dc) => {
    const legal = env.legalActions();
    if (JSON.stringify(legal) !== JSON.stringify(dc.legal)) bad.push(['legal', n, legal, dc.legal]);
    const x = E.encode(env, env.toAct, dc.eq);
    if (x.some((v, i) => Math.abs(v - dc.obs[i]) > 1e-6)) bad.push(['obs', n]);
    const p = policy.forward(x, legal).probs;
    if (p.some((v, i) => Math.abs(v - dc.probs[i]) > 1e-4)) bad.push(['probs', n, p, dc.probs]);
    env.step(dc.action);
  });
  if (!env.done || env.payoff[0] !== h.payoff[0]) bad.push(['payoff', n, env.payoff, h.payoff]);
});
console.log(JSON.stringify(bad.slice(0, 5)));
"""


@unittest.skipUnless(shutil.which('node'), 'node is not installed')
class TestParity(unittest.TestCase):
    def test_js_matches_python(self):
        nrng = np.random.default_rng(0)
        shapes = {'body.0.weight': (256, OBS_SIZE), 'body.0.bias': (256,), 'body.2.weight': (256, 256),
                  'body.2.bias': (256,), 'pi.weight': (5, 256), 'pi.bias': (5,), 'v.weight': (1, 256), 'v.bias': (1,)}
        w = {k: (nrng.standard_normal(s) * 0.2).astype(np.float32) for k, s in shapes.items()}
        rng = random.Random(0)
        env, agent = HeadsUpEnv(), RandomAgent()
        hands = []
        for n in range(400):
            deck = list(range(52))
            rng.shuffle(deck)
            stacks = [rng.randint(1, 300) * env.bb, rng.randint(1, 300) * env.bb]
            env.reset(deck, n % 2, stacks=stacks)
            decisions = []
            while not env.done:
                eq = rng.random()
                x = encode(env, env.to_act, eq)
                p, _ = forward(w, x, legal_mask(env))
                a = agent.act(env, env.to_act, None, rng)
                decisions.append({'legal': env.legal_actions(), 'eq': eq, 'obs': x.tolist(), 'probs': p.tolist(), 'action': a})
                env.step(a)
            hands.append({'deck': deck, 'button': n % 2, 'stacks': stacks, 'decisions': decisions, 'payoff': env.payoff})
        evals = []
        for _ in range(3000):
            cs = rng.sample(range(52), rng.choice([5, 6, 7]))
            evals.append([cs, evaluate(cs)])
        data = {'hands': hands, 'evals': evals,
                'weights': {k: {'shape': list(v.shape), 'data': v.flatten().tolist()} for k, v in w.items()}}

        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            json.dump(data, f)
        try:
            out = subprocess.run(['node', '-e', JS_CHECK, os.path.abspath(ENGINE), f.name],
                                 capture_output=True, text=True, check=True).stdout
        finally:
            os.unlink(f.name)
        self.assertEqual(json.loads(out), [])


if __name__ == '__main__':
    unittest.main()
