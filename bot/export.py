"""Export a checkpoint for the browser: network weights plus the preflop equity table, in one JSON file.

  python export.py --ckpt ckpt/latest.pt --out ../bot-model.json
"""
import argparse
import base64
import json

import numpy as np
import torch

from poker.cards import PREFLOP_FILE
from poker.env import ACTION_NAMES
from poker.features import OBS_SIZE


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', default='ckpt/latest.pt')
    p.add_argument('--out', default='../bot-model.json')
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    with open(PREFLOP_FILE) as f:
        preflop = json.load(f)
    out = {
        'obs_size': OBS_SIZE,
        'actions': ACTION_NAMES,
        'iter': ck['iter'],
        'hands': ck['hands'],
        'preflop': preflop,
        # little-endian float32, base64; torch Linear weights are [out, in]
        'weights': {k: {'shape': list(v.shape),
                        'b64': base64.b64encode(np.asarray(v.numpy(), dtype='<f4').tobytes()).decode()}
                    for k, v in ck['model'].items()},
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    print('wrote', args.out)


if __name__ == '__main__':
    main()
