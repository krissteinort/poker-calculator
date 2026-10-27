"""Measure a trained bot with duplicate matches: the bot plays against copies of an opponent, and every deck is
replayed with the bot in each seat, which cancels out most of the luck.

  python evaluate.py --ckpt ckpt/latest.pt --vs equity call raise random --players 2 6
  python evaluate.py --ckpt ckpt/latest.pt --vs old_ckpt/latest.pt
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import multiprocessing as mp

import torch

from poker.agents import BASELINES
from poker.model import PolicyNet, numpy_weights
from poker.play import duplicate_match


def load_policy(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = PolicyNet(ck.get('hidden', 256))
    model.load_state_dict(ck['model'])
    return ('policy', numpy_weights(model))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', default='ckpt/latest.pt')
    p.add_argument('--vs', nargs='+', default=['equity', 'call', 'raise', 'random'],
                   help='baseline names (%s) or checkpoint paths' % ', '.join(BASELINES))
    p.add_argument('--hands', type=int, default=40000, help='hands per matchup')
    p.add_argument('--players', type=int, nargs='+', default=[2, 6], help='table sizes to test')
    p.add_argument('--stack-bb', type=float, nargs='+', default=[100], help='stack depths to test, in big blinds')
    p.add_argument('--workers', type=int, default=os.cpu_count())
    args = p.parse_args()

    me = load_policy(args.ckpt)
    with mp.get_context('spawn').Pool(args.workers) as pool:
        for players in args.players:
            for depth in args.stack_bb:
                for opp in args.vs:
                    spec = opp if opp in BASELINES else load_policy(opp)
                    mbb, ci = duplicate_match(pool, me, spec, args.hands // players, seed=1, chunks=args.workers * 4,
                                              stack_bb=depth, players=players)
                    print(f'{players} players, {depth:g}bb, vs {opp:>12}: {mbb:+8.0f} mbb/hand  (95% CI ±{ci:.0f})')


if __name__ == '__main__':
    main()
