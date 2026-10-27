# Poker bot

A no-limit hold'em bot for tables of 2–6 players, trained by self-play with PPO.

- **Game** ([poker/env.py](poker/env.py)): standard blinds and turn order, with side pots for unequal all-ins. Every decision is one of five actions: fold, check/call, raise ½ pot, raise pot, all-in.
- **Training tables:** each hand draws a table size of 2–6 players (`--min-players`/`--max-players`) and stacks of 10–250 big blinds (`--min-stack-bb`/`--max-stack-bb`). `--hu-frac 0.3` sends an extra 30% of games to heads-up tables.
- **What the network sees** ([poker/features.py](poker/features.py)):
  - the player's cards and the board;
  - **equity against as many random hands as there are opponents left**: a precomputed table before the flop, Monte Carlo after it;
  - table size, players still in the hand, and the player's seat relative to the button;
  - pot, stack and price-to-call figures;
  - the current hand type;
  - a summary of the betting so far;
  - for each opponent, in seat order: still in or folded, all-in, stack, bets and raises.
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

- `evaluate.py` uses duplicate matches: the bot plays against copies of one opponent, and each deck is replayed with the bot in every seat. This cancels out most of the luck.
- Beating `call`, `raise` and `equity` by a wide margin is the first milestone.
- To check that newer checkpoints still improve, play them against older ones with `--vs path/to/old.pt`.
- Matches use 100bb stacks at 2 and 6 players by default. Change them with `--stack-bb 20 100 250` and `--players 2 3 6`. Training's own checkpoint evaluations use 100bb at the smallest and largest table size.

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

## Playing it in the browser

[play.html](../play.html) lets visitors play against 1–5 bots. It runs entirely in the browser:

- [play-engine.js](../play-engine.js) is a JS copy of the game rules, the features and the network.
- [bot-model.json](../bot-model.json) is the multi-player bot, used at 3–6 players: the trained weights and the preflop equity table.
- [bot-model-hu.json](../bot-model-hu.json) is used for heads-up games. It's an older bot trained only on 2-player hands (6M of them), with a smaller input layout (`encodeHU` in play-engine.js). It still beats the multi-player bot heads-up, especially with deep stacks. The Python code that trained it is in git history (commit `1eb7a7f`).

To ship a newer multi-player bot:

```bash
.venv/bin/python export.py --ckpt ckpt/latest.pt --out ../bot-model.json
```

Then commit and push `bot-model.json`. The site uses the new bot as soon as GitHub Pages redeploys. If the multi-player bot ever beats the heads-up bot heads-up, export it to `bot-model-hu.json` as well. The page picks the input layout from the file's `obs_size`.

If you change the rules or the features in Python, change `play-engine.js` to match. [tests/test_parity.py](tests/test_parity.py) fails if the two disagree. It needs `node` installed and is skipped without it.

To try the page locally, serve the repo root with `python3 -m http.server` and open `http://localhost:8000/play.html`. The page loads the bot with `fetch`, which doesn't work when the file is opened directly.
