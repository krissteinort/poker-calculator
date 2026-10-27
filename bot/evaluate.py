"""Measure a trained bot with duplicate matches (same cards played from both seats to cancel out luck).

  python evaluate.py --ckpt ckpt/latest.pt --vs equity call raise random
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
    p.add_argument('--pairs', type=int, default=20000)
    p.add_argument('--workers', type=int, default=os.cpu_count())
    args = p.parse_args()

    me = load_policy(args.ckpt)
    with mp.get_context('spawn').Pool(args.workers) as pool:
        for opp in args.vs:
            spec = opp if opp in BASELINES else load_policy(opp)
            mbb, ci = duplicate_match(pool, me, spec, args.pairs, seed=1, chunks=args.workers * 4)
            print(f'vs {opp:>12}: {mbb:+8.0f} mbb/hand  (95% CI ±{ci:.0f}, {2 * args.pairs:,} hands)')


if __name__ == '__main__':
    main()
