"""Export a checkpoint's weights to JSON so the calculator page can run the bot in the browser.

  python export.py --ckpt ckpt/latest.pt --out ../bot-model.json
"""
import argparse
import json

import torch

from poker.env import ACTION_NAMES
from poker.features import OBS_SIZE


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', default='ckpt/latest.pt')
    p.add_argument('--out', default='../bot-model.json')
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    out = {
        'obs_size': OBS_SIZE,
        'actions': ACTION_NAMES,
        'iter': ck['iter'],
        'hands': ck['hands'],
        'weights': {k: {'shape': list(v.shape), 'data': [round(x, 5) for x in v.flatten().tolist()]}
                    for k, v in ck['model'].items()},
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    print('wrote', args.out)


if __name__ == '__main__':
    main()
