# Poker bot

A heads-up (two-player) no-limit hold'em bot trained by self-play with PPO.

- **Game:** 100 big blind stacks. Every decision is one of five actions: fold, check/call, raise ½ pot, raise pot, all-in.
- **What the network sees** ([poker/features.py](poker/features.py)):
  - the player's cards and the board;
  - **equity against a random hand** (a precomputed table before the flop, Monte Carlo after it);
  - pot, stack and price-to-call figures;
  - the current hand type;
  - a summary of the betting so far.
- **Network:** a 2×256 MLP that outputs both the action probabilities and a value estimate.
- **Training:** the bot plays against its current self, past snapshots of itself, and a few simple bots. Hands are simulated on CPU worker processes with numpy, and the update step uses a GPU if there is one.

## Local setup

```bash
cd bot
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest tests.test_poker      # evaluator, rules, chip conservation
.venv/bin/python train.py --hours 0.5              # writes ckpt/latest.pt and ckpt/log.jsonl
.venv/bin/python evaluate.py --ckpt ckpt/latest.pt # duplicate matches vs the simple bots
```

Re-running `train.py` with the same `--ckpt-dir` resumes from the last checkpoint.

### Reading the results

Results are in **mbb/hand**: thousandths of a big blind won per hand.

- `evaluate.py` uses duplicate matches: each deck is played twice with the seats swapped, which cancels out most of the luck.
- Beating `call`, `raise` and `equity` by a wide margin is the first milestone.
- To check that newer checkpoints still improve, play them against older ones with `--vs path/to/old.pt`.

The simple bots it is measured against (in [poker/agents.py](poker/agents.py)):

| Bot | Plays |
|---|---|
| `random` | Picks any legal action |
| `call` | Never folds, never raises |
| `raise` | Raises the pot every time |
| `equity` | Bets strong hands and calls when the pot odds are right |

## RunPod

Most of the work is simulating hands, so **CPU cores matter more than the GPU**. A CPU pod, or the cheapest GPU pod with plenty of vCPUs, is the right choice.

1. Create a pod from a PyTorch template.
   - Use Community Cloud, which is cheaper.
   - Attach a volume at `/workspace` so checkpoints survive the pod being stopped.
2. In the pod's terminal:
   ```bash
   cd /workspace && git clone https://github.com/krissteinort/poker-calculator.git
   cd poker-calculator/bot && HOURS=4 bash runpod/start.sh --workers $(nproc)
   ```
   The script:
   - runs the tests;
   - trains for `HOURS`;
   - prints an evaluation;
   - **stops the pod** so billing ends. Set `STOP_WHEN_DONE=0` to keep the pod running.

   To keep training running if your browser disconnects, start it inside `tmux`.
3. Download `/workspace/poker-ckpt/latest.pt` from the pod. Use `runpodctl send`, or the file browser in Jupyter.

## Using the bot in the calculator

```bash
.venv/bin/python export.py --ckpt ckpt/latest.pt --out ../bot-model.json
```

This writes the weights as JSON. The calculator page can then run the network in plain JS: the feature encoding is small enough to re-implement, and the equity engine already exists there.
