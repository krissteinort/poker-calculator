"""PPO self-play training for the heads-up bot.

Resumable: re-running with the same --ckpt-dir picks up from the last checkpoint.
Hands are simulated on CPU worker processes; the network update uses a GPU if there is one.
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')  # workers are single-threaded; the trainer sets torch threads itself

import argparse
import json
import multiprocessing as mp
import random
import time

import numpy as np
import torch

from poker.model import PolicyNet, numpy_weights
from poker.play import duplicate_match, rollout


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt-dir', default='ckpt')
    p.add_argument('--hours', type=float, default=1.0, help='stop after this much wall-clock time')
    p.add_argument('--max-iters', type=int, default=0, help='0 = no limit')
    p.add_argument('--workers', type=int, default=os.cpu_count())
    p.add_argument('--hands-per-iter', type=int, default=8000)
    p.add_argument('--hands-per-task', type=int, default=200)
    p.add_argument('--min-stack-bb', type=float, default=10, help='training stacks are drawn between these')
    p.add_argument('--max-stack-bb', type=float, default=250)
    p.add_argument('--min-players', type=int, default=2, help='table sizes are drawn between these')
    p.add_argument('--max-players', type=int, default=6)
    p.add_argument('--self-seat-frac', type=float, default=0.3,
                   help='vs past selves or simple bots, the share of the other seats played by the current network')
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--minibatch', type=int, default=2048)
    p.add_argument('--clip', type=float, default=0.2)
    p.add_argument('--ent', type=float, default=0.01, help='entropy bonus (keeps the strategy mixed)')
    p.add_argument('--vf', type=float, default=0.5)
    p.add_argument('--gae-lambda', type=float, default=0.95)
    p.add_argument('--self-play-frac', type=float, default=0.5, help='share of games vs the current network')
    p.add_argument('--baseline-frac', type=float, default=0.1, help='share of games vs the simple bots')
    p.add_argument('--snapshot-every', type=int, default=10, help='iterations between adding a past self to the pool')
    p.add_argument('--pool-size', type=int, default=20)
    p.add_argument('--eval-every', type=int, default=25)
    p.add_argument('--eval-hands', type=int, default=4000, help='hands per matchup in the periodic evaluations')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def advantages(rew, val, done, lam):
    """GAE with gamma = 1 (a hand is short and only the final chip result matters)."""
    adv = np.zeros_like(rew)
    last = 0.0
    for t in range(len(rew) - 1, -1, -1):
        next_v = 0.0 if done[t] else val[t + 1]
        if done[t]:
            last = 0.0
        delta = rew[t] + next_v - val[t]
        last = delta + lam * last
        adv[t] = last
    return adv, adv + val


def ppo_update(model, opt, batch, args, device):
    t = {k: torch.from_numpy(batch[k]).to(device) for k in ('obs', 'mask', 'act', 'logp', 'adv', 'ret')}
    adv = (t['adv'] - t['adv'].mean()) / (t['adv'].std() + 1e-8)
    n = len(adv)
    stats = []
    for _ in range(args.epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, args.minibatch):
            i = perm[s:s + args.minibatch]
            logits, v = model(t['obs'][i], t['mask'][i])
            logp_all = torch.log_softmax(logits, -1)
            logp = logp_all.gather(1, t['act'][i, None]).squeeze(1)
            ratio = torch.exp(logp - t['logp'][i])
            pg = -torch.min(ratio * adv[i], ratio.clamp(1 - args.clip, 1 + args.clip) * adv[i]).mean()
            vl = ((v - t['ret'][i]) ** 2).mean()
            ent = -(logp_all.exp() * logp_all).sum(-1).mean()
            loss = pg + args.vf * vl - args.ent * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            stats.append((pg.item(), vl.item(), ent.item(), (t['logp'][i] - logp).mean().item()))
    return dict(zip(('policy_loss', 'value_loss', 'entropy', 'kl'), np.mean(stats, axis=0).tolist()))


def save(path, state):
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic, so a killed pod never leaves a half-written checkpoint


def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, 'latest.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(min(8, os.cpu_count()))
    rng = random.Random(args.seed)

    model = PolicyNet(args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    it, hands, pool_snaps = 0, 0, []
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        it, hands, pool_snaps = ck['iter'], ck['hands'], ck['pool']
        rng = random.Random(args.seed + it)
        print(f'resumed from iteration {it} ({hands:,} hands)')

    log = open(os.path.join(args.ckpt_dir, 'log.jsonl'), 'a')
    workers = mp.get_context('spawn').Pool(args.workers)
    print(f'training on {device} with {args.workers} workers; log in {args.ckpt_dir}/log.jsonl')
    start = time.time()

    def checkpoint():
        save(ckpt_path, {'model': model.state_dict(), 'opt': opt.state_dict(), 'iter': it, 'hands': hands,
                         'pool': pool_snaps, 'hidden': args.hidden})

    while time.time() - start < args.hours * 3600 and (not args.max_iters or it < args.max_iters):
        t0 = time.time()
        w = numpy_weights(model)
        tasks = []
        for _ in range(max(1, args.hands_per_iter // args.hands_per_task)):
            r = rng.random()
            self_frac = args.self_seat_frac
            if r < args.baseline_frac:
                opp = rng.choice(['call', 'raise', 'equity', 'random'])
            elif r < args.baseline_frac + args.self_play_frac or not pool_snaps:
                opp, self_frac = ('policy', w), 1.0  # pure self-play: every seat is the learner
            else:
                opp = ('policy', rng.choice(pool_snaps))
            tasks.append((w, opp, args.hands_per_task, rng.randrange(2 ** 31), (args.min_stack_bb, args.max_stack_bb),
                          (args.min_players, args.max_players), self_frac))
        parts = workers.map(rollout, tasks)
        batch = {k: np.concatenate([p[k] for p in parts]) for k in ('obs', 'mask', 'act', 'logp', 'val', 'rew', 'done')}
        batch['adv'], batch['ret'] = advantages(batch['rew'], batch['val'], batch['done'], args.gae_lambda)
        sim_time = time.time() - t0

        model.train()
        stats = ppo_update(model, opt, batch, args, device)
        it += 1
        n_hands = sum(p['hands'] for p in parts)
        hands += n_hands
        if it % args.snapshot_every == 0:
            pool_snaps.append(numpy_weights(model))
            pool_snaps = pool_snaps[-args.pool_size:]

        row = {'iter': it, 'hands': hands, 'decisions': len(batch['act']),
               'bb_per_hand': sum(p['winnings'] for p in parts) / max(1, sum(p['seats'] for p in parts)),
               'sim_s': round(sim_time, 1), 'total_s': round(time.time() - t0, 1), **stats}
        if it % args.eval_every == 0:
            w = numpy_weights(model)
            for players in sorted({args.min_players, args.max_players}):
                for opp in ('equity', 'call'):
                    mbb, ci = duplicate_match(workers, ('policy', w), opp, args.eval_hands // players, it, args.workers,
                                              players=players)
                    row[f'{players}p_vs_{opp}_mbb'] = round(mbb)
                    row[f'{players}p_vs_{opp}_ci'] = round(ci)
        log.write(json.dumps(row) + '\n')
        log.flush()
        print(' '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}' for k, v in row.items()), flush=True)
        checkpoint()

    checkpoint()
    workers.close()
    print(f'done: {it} iterations, {hands:,} hands, checkpoint at {ckpt_path}')


if __name__ == '__main__':
    main()
