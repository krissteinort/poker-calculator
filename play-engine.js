// No-limit hold'em engine for play.html (2-6 players): a JS mirror of bot/poker (cards.py, env.py, features.py,
// agents.forward). Keep it in sync with the Python; bot/tests/test_parity.py checks that the two agree.
(function (root) {
  var FOLD = 0, CALL = 1, HALF_POT = 2, POT = 3, ALL_IN = 4;
  var BOARD_COUNT = [0, 3, 4, 5, 5];
  var MAX_PLAYERS = 6, OPP_FEATURES = 7;
  var OBS_SIZE = 52 + 52 + 1 + 4 + 5 + 5 + MAX_PLAYERS + 1 + 5 + 9 + 16 + (MAX_PLAYERS - 1) * OPP_FEATURES;
  var CHIP_SCALE_BB = 100;  // chip amounts are fed to the network in units of 100 big blinds
  var CATEGORIES = ['High Card', 'Pair', 'Two Pair', 'Three of a Kind', 'Straight',
    'Flush', 'Full House', 'Four of a Kind', 'Straight Flush'];

  // ---- evaluator (card = rank * 4 + suit) ----
  var BITS = [], STRAIGHT = [], RANKLIST = [], TOP5 = [];
  for (var m = 0; m < 8192; m++) {
    var list = [];
    for (var r = 12; r >= 0; r--) if ((m >> r) & 1) list.push(r);
    RANKLIST.push(list);
    BITS.push(list.length);
    var w = (m << 1) | ((m >> 12) & 1), sh = -1;
    for (var h = 13; h >= 4; h--) if (((w >> (h - 4)) & 31) === 31) { sh = h - 1; break; }
    STRAIGHT.push(sh);
    var v = 0, n = Math.min(5, list.length);
    for (var i = 0; i < n; i++) v = (v << 4) | list[i];
    TOP5.push(v << (4 * (5 - n)));
  }

  function evaluate(cs) {
    var rc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], sms = [0, 0, 0, 0], mask = 0, i, r;
    for (i = 0; i < cs.length; i++) {
      r = cs[i] >> 2;
      rc[r]++;
      sms[cs[i] & 3] |= 1 << r;
      mask |= 1 << r;
    }
    var flush = -1;
    for (i = 0; i < 4; i++) {
      if (BITS[sms[i]] >= 5) {
        if (STRAIGHT[sms[i]] >= 0) return (8 << 20) | (STRAIGHT[sms[i]] << 16);
        flush = (5 << 20) | TOP5[sms[i]];
        break;
      }
    }
    if (BITS[mask] === cs.length) {
      if (flush >= 0) return flush;
      if (STRAIGHT[mask] >= 0) return (4 << 20) | (STRAIGHT[mask] << 16);
      return TOP5[mask];
    }
    var quad = -1, t1 = -1, t2 = -1, p1 = -1, p2 = -1;
    for (r = 12; r >= 0; r--) {
      var c = rc[r];
      if (c < 2) continue;
      if (c === 4) quad = r;
      else if (c === 3) { if (t1 < 0) t1 = r; else if (t2 < 0) t2 = r; }
      else if (p1 < 0) p1 = r;
      else if (p2 < 0) p2 = r;
    }
    var k;
    if (quad >= 0) return (7 << 20) | (quad << 16) | (RANKLIST[mask & ~(1 << quad)][0] << 12);
    if (t1 >= 0 && (t2 >= 0 || p1 >= 0)) return (6 << 20) | (t1 << 16) | (Math.max(t2, p1) << 12);
    if (flush >= 0) return flush;
    if (STRAIGHT[mask] >= 0) return (4 << 20) | (STRAIGHT[mask] << 16);
    if (t1 >= 0) { k = RANKLIST[mask & ~(1 << t1)]; return (3 << 20) | (t1 << 16) | (k[0] << 12) | (k[1] << 8); }
    if (p2 >= 0) { k = RANKLIST[mask & ~(1 << p1) & ~(1 << p2)]; return (2 << 20) | (p1 << 16) | (p2 << 12) | (k[0] << 8); }
    k = RANKLIST[mask & ~(1 << p1)];
    return (1 << 20) | (p1 << 16) | (k[0] << 12) | (k[1] << 8) | (k[2] << 4);
  }

  // Monte Carlo equity (share of the pot won, ties split) vs `opponents` random hands.
  function equityMC(hole, board, samples, opponents) {
    var known = {}, deck = [], i, j, k2 = 2 * (opponents || 1);
    hole.concat(board).forEach(function (c) { known[c] = true; });
    for (i = 0; i < 52; i++) if (!known[i]) deck.push(i);
    var need = 5 - board.length, total = 0;
    for (var s = 0; s < samples; s++) {
      for (i = 0; i < need + k2; i++) {
        j = i + Math.floor(Math.random() * (deck.length - i));
        var t = deck[i]; deck[i] = deck[j]; deck[j] = t;
      }
      var b = board.concat(deck.slice(k2, k2 + need)), me = evaluate(hole.concat(b)), ties = 0, lost = false;
      for (var o = 0; o < k2; o += 2) {
        var sc = evaluate([deck[o], deck[o + 1]].concat(b));
        if (sc > me) { lost = true; break; }
        if (sc === me) ties++;
      }
      if (!lost) total += 1 / (ties + 1);
    }
    return total / samples;
  }

  var RANKS = '23456789TJQKA';
  function preflopClass(hole) {
    var a = hole[0], b = hole[1];
    if ((b >> 2) > (a >> 2)) { var t = a; a = b; b = t; }
    var ra = RANKS[a >> 2], rb = RANKS[b >> 2];
    if (ra === rb) return ra + rb;
    return ra + rb + ((a & 3) === (b & 3) ? 's' : 'o');
  }

  // ---- game (mirrors env.py) ----
  function Env(nPlayers, stackBB, bb) {
    this.n = nPlayers || 2;
    this.bb = bb || 2;
    this.startStack = (stackBB || 100) * this.bb;
  }
  // stacks: starting chips per seat (its length sets the number of players); each must cover a big blind.
  Env.prototype.reset = function (deck, button, stacks) {
    if (stacks) this.n = stacks.length;
    var n = this.n, i;
    this.hole = [];
    for (i = 0; i < n; i++) this.hole.push(deck.slice(2 * i, 2 * i + 2));
    this.fullBoard = deck.slice(2 * n, 2 * n + 5);
    this.button = button;
    this.stack = stacks ? stacks.slice() : fill(n, this.startStack);
    this.bet = fill(n, 0);
    this.contrib = fill(n, 0);
    this.folded = fill(n, false);
    this.acted = fill(n, false);
    this.street = 0;
    this.history = [[], [], [], []];
    this.done = false;
    this.payoff = null;
    this.winnings = null;  // chips each seat took from the pot
    this.sbSeat = n === 2 ? button : (button + 1) % n;
    this.bbSeat = (this.sbSeat + 1) % n;
    this._put(this.sbSeat, this.bb / 2);
    this._put(this.bbSeat, this.bb);
    this.lastRaise = this.bb;
    this.toAct = this._nextNeeding(this.bbSeat);
  };
  function fill(n, v) { var a = []; for (var i = 0; i < n; i++) a.push(v); return a; }
  function sum(a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return s; }
  Env.prototype.board = function () { return this.fullBoard.slice(0, BOARD_COUNT[this.street]); };
  Env.prototype.pot = function () { return sum(this.contrib); };
  Env.prototype.toCall = function (seat) { return Math.max.apply(null, this.bet) - this.bet[seat]; };
  Env.prototype.canAct = function (seat) { return !this.folded[seat] && this.stack[seat] > 0; };
  Env.prototype.inHand = function () {
    var out = [];
    for (var s = 0; s < this.n; s++) if (!this.folded[s]) out.push(s);
    return out;
  };
  Env.prototype._put = function (seat, amount) {
    amount = Math.min(amount, this.stack[seat]);
    this.stack[seat] -= amount;
    this.bet[seat] += amount;
    this.contrib[seat] += amount;
  };
  Env.prototype._nextNeeding = function (after) {
    var mb = Math.max.apply(null, this.bet);
    for (var i = 1; i <= this.n; i++) {
      var s = (after + i) % this.n;
      if (this.canAct(s) && (!this.acted[s] || this.bet[s] < mb)) return s;
    }
    return null;
  };
  Env.prototype.raiseAmount = function (seat, frac) {
    var tc = this.toCall(seat);
    return tc + Math.max(Math.floor(frac * (this.pot() + tc)), this.lastRaise, this.bb);
  };
  Env.prototype.legalActions = function () {
    var seat = this.toAct, tc = this.toCall(seat), acts = tc > 0 ? [FOLD] : [], others = false;
    acts.push(CALL);
    for (var o = 0; o < this.n; o++) if (o !== seat && this.canAct(o)) others = true;
    if (this.stack[seat] > tc && others) {
      if (this.raiseAmount(seat, 0.5) < this.stack[seat]) acts.push(HALF_POT);
      if (this.raiseAmount(seat, 1) < this.stack[seat]) acts.push(POT);
      acts.push(ALL_IN);
    }
    return acts;
  };
  // Chips the player to act would add for action a (for labels).
  Env.prototype.amountFor = function (a) {
    var seat = this.toAct;
    if (a === FOLD) return 0;
    if (a === CALL) return Math.min(this.toCall(seat), this.stack[seat]);
    if (a === ALL_IN) return this.stack[seat];
    return this.raiseAmount(seat, a === HALF_POT ? 0.5 : 1);
  };
  Env.prototype.step = function (a) {
    var seat = this.toAct;
    if (this.done || this.legalActions().indexOf(a) < 0) throw new Error('illegal action ' + a);
    this.history[this.street].push([seat, a]);
    if (a === FOLD) {
      this.folded[seat] = true;
      var left = this.inHand();
      if (left.length === 1) {
        var win = fill(this.n, 0);
        win[left[0]] = this.pot();
        return this._finish(win);
      }
    } else if (a === CALL) {
      this._put(seat, this.toCall(seat));
      this.acted[seat] = true;
    } else {
      var amount = a === ALL_IN ? this.stack[seat] : this.raiseAmount(seat, a === HALF_POT ? 0.5 : 1);
      var raiseBy = this.bet[seat] + amount - Math.max.apply(null, this.bet);
      this._put(seat, amount);
      this.lastRaise = Math.max(this.lastRaise, raiseBy);
      this.acted = fill(this.n, false);
      this.acted[seat] = true;
    }
    var next = this._nextNeeding(seat);
    if (next === null) this._endRound();
    else this.toAct = next;
  };
  Env.prototype._endRound = function () {
    var canAct = 0;
    for (var s = 0; s < this.n; s++) if (this.canAct(s)) canAct++;
    if (canAct <= 1 || this.street === 3) {
      this.street = 4;
      return this._showdown();
    }
    this.street++;
    this.bet = fill(this.n, 0);
    this.acted = fill(this.n, false);
    this.lastRaise = this.bb;
    this.toAct = this._nextNeeding(this.button);
  };
  // Split the main pot and any side pots between the best hands eligible for each.
  Env.prototype._showdown = function () {
    var self = this, live = this.inHand(), score = {}, win = fill(this.n, 0), prev = 0;
    live.forEach(function (s) { score[s] = evaluate(self.hole[s].concat(self.fullBoard)); });
    var levels = live.map(function (s) { return self.contrib[s]; })
      .filter(function (v, i, a) { return a.indexOf(v) === i; }).sort(function (a, b) { return a - b; });
    this.pots = [];  // for the UI: {amount, winners}
    levels.forEach(function (level) {
      var part = 0;
      self.contrib.forEach(function (c) { part += Math.min(c, level) - Math.min(c, prev); });
      var eligible = live.filter(function (s) { return self.contrib[s] >= level; });
      var best = Math.max.apply(null, eligible.map(function (s) { return score[s]; }));
      var winners = eligible.filter(function (s) { return score[s] === best; });
      var share = Math.floor(part / winners.length), odd = part - share * winners.length;
      // odd chips go to the winners closest to the left of the button
      winners.sort(function (a, b) { return mod(a - self.button - 1, self.n) - mod(b - self.button - 1, self.n); });
      winners.forEach(function (s, k) { win[s] += share + (k < odd ? 1 : 0); });
      self.pots.push({ amount: part, winners: winners, eligible: eligible.length });
      prev = level;
    });
    this._finish(win);
  };
  function mod(a, n) { return ((a % n) + n) % n; }
  Env.prototype._finish = function (win) {
    this.done = true;
    this.winnings = win;
    var self = this;
    this.payoff = win.map(function (w, s) { return (w - self.contrib[s]) / self.bb; });
  };

  // ---- features (mirrors features.py) ----
  function encode(env, seat, eq) {
    var x = new Float32Array(OBS_SIZE), board = env.board(), n = env.n, i, o, k;
    env.hole[seat].forEach(function (c) { x[c] = 1; });
    board.forEach(function (c) { x[52 + c] = 1; });
    i = 104;
    x[i++] = eq;
    x[i + env.street] = 1; i += 4;
    x[i + n - 2] = 1; i += 5;
    x[i + env.inHand().length - 2] = 1; i += 5;
    x[i + mod(seat - env.button, n)] = 1; i += MAX_PLAYERS;
    var waiting = 0;
    for (o = 0; o < n; o++) if (o !== seat && env.canAct(o) && !env.acted[o]) waiting++;
    x[i++] = waiting / 5;

    var s0 = CHIP_SCALE_BB * env.bb, tc = env.toCall(seat), pot = env.pot();
    x[i] = env.stack[seat] / s0;
    x[i + 1] = pot / (2 * s0);
    x[i + 2] = tc / s0;
    x[i + 3] = tc / (pot + tc);
    x[i + 4] = env.bet[seat] / s0;
    i += 5;
    var cards = env.hole[seat].concat(board);
    x[i + (cards.length >= 5 ? evaluate(cards) >> 20 : ((cards[0] >> 2) === (cards[1] >> 2) ? 1 : 0))] = 1;
    i += 9;
    for (var st = 0; st < 4; st++) {
      env.history[st].forEach(function (h) {
        if (h[1] === FOLD) return;
        x[i + st * 4 + (h[0] === seat ? 0 : 2) + (h[1] === CALL ? 1 : 0)] += 0.25;
      });
    }
    i += 16;
    for (k = 1; k < MAX_PLAYERS; k++) {
      if (k < n) {
        o = (seat + k) % n;
        x[i] = 1;
        x[i + 1] = env.folded[o] ? 0 : 1;
        x[i + 2] = !env.folded[o] && env.stack[o] === 0 ? 1 : 0;
        x[i + 3] = env.stack[o] / s0;
        x[i + 4] = env.bet[o] / s0;
        x[i + 5] = env.contrib[o] / s0;
        var raises = 0;
        env.history[env.street].forEach(function (h) { if (h[0] === o && h[1] > CALL) raises++; });
        x[i + 6] = 0.25 * raises;
      }
      i += OPP_FEATURES;
    }
    return x;
  }

  // The heads-up-only bot (bot-model-hu.json) was trained on this older, 2-player layout.
  var HU_OBS_SIZE = 52 + 52 + 1 + 4 + 1 + 5 + 9 + 16;
  function encodeHU(env, seat, eq) {
    var x = new Float32Array(HU_OBS_SIZE), board = env.board(), i;
    env.hole[seat].forEach(function (c) { x[c] = 1; });
    board.forEach(function (c) { x[52 + c] = 1; });
    i = 104;
    x[i++] = eq;
    x[i + env.street] = 1; i += 4;
    x[i++] = env.button === seat ? 1 : 0;
    var s0 = CHIP_SCALE_BB * env.bb, tc = env.toCall(seat), pot = env.pot();
    x[i] = pot / (2 * s0);
    x[i + 1] = env.stack[seat] / s0;
    x[i + 2] = env.stack[1 - seat] / s0;
    x[i + 3] = tc / s0;
    x[i + 4] = tc / (pot + tc);
    i += 5;
    var cards = env.hole[seat].concat(board);
    x[i + (cards.length >= 5 ? evaluate(cards) >> 20 : ((cards[0] >> 2) === (cards[1] >> 2) ? 1 : 0))] = 1;
    i += 9;
    for (var st = 0; st < 4; st++) {
      env.history[st].forEach(function (h) {
        if (h[1] === FOLD) return;
        x[i + st * 4 + (h[0] === seat ? 0 : 1) + (h[1] === CALL ? 2 : 0)] += 0.25;
      });
    }
    return x;
  }

  function Layer(w, b) { this.w = w; this.b = b; }
  Layer.prototype.apply = function (x, relu) {
    var out = this.b.data.length, inp = x.length, y = new Float64Array(out), W = this.w.data;
    for (var o = 0; o < out; o++) {
      var s = this.b.data[o], row = o * inp;
      for (var i = 0; i < inp; i++) s += W[row + i] * x[i];
      y[o] = relu && s < 0 ? 0 : s;
    }
    return y;
  };

  // weights: {name: {shape, data}} with torch's [out, in] layout.
  function Policy(weights) {
    var g = function (n) { return new Layer(weights[n + '.weight'], weights[n + '.bias']); };
    this.l1 = g('body.0'); this.l2 = g('body.2'); this.pi = g('pi'); this.v = g('v');
  }
  Policy.prototype.forward = function (x, legal) {
    var h = this.l2.apply(this.l1.apply(x, true), true);
    var logits = this.pi.apply(h, false), max = -Infinity, probs = [], total = 0, a;
    for (a = 0; a < logits.length; a++) {
      if (legal.indexOf(a) < 0) logits[a] = -1e9;
      if (logits[a] > max) max = logits[a];
    }
    for (a = 0; a < logits.length; a++) { probs.push(Math.exp(logits[a] - max)); total += probs[a]; }
    return { probs: probs.map(function (p) { return p / total; }), value: this.v.apply(h, false)[0] };
  };

  function decodeModel(json) {
    var weights = {};
    Object.keys(json.weights).forEach(function (k) {
      var bin = atob(json.weights[k].b64), bytes = new Uint8Array(bin.length);
      for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      weights[k] = { shape: json.weights[k].shape, data: new Float32Array(bytes.buffer) };
    });
    return weights;
  }

  // The bots: one network shared by every bot seat. Equity is cached per (seat, street, opponents left) like
  // EquityCache in features.py. A heads-up model (obs_size 140) uses encodeHU and a preflop table of plain numbers.
  function Bot(json) {
    this.policy = new Policy(decodeModel(json));
    this.preflop = json.preflop;
    this.headsUp = json.obs_size === HU_OBS_SIZE;
    this.cache = {};
  }
  Bot.prototype.newHand = function () { this.cache = {}; };
  Bot.prototype.equity = function (env, seat) {
    var opp = env.inHand().length - 1, key = seat + ':' + env.street + ':' + opp;
    if (!(key in this.cache)) {
      var board = env.board();
      var pre = board.length ? null : this.preflop[preflopClass(env.hole[seat])];
      this.cache[key] = board.length ? equityMC(env.hole[seat], board, 128, opp) : this.headsUp ? pre : pre[opp - 1];
    }
    return this.cache[key];
  };
  Bot.prototype.act = function (env, seat) {
    var x = (this.headsUp ? encodeHU : encode)(env, seat, this.equity(env, seat));
    var out = this.policy.forward(x, env.legalActions());
    var r = Math.random(), a = 0;
    for (; a < out.probs.length - 1; a++) { r -= out.probs[a]; if (r < 0) break; }
    while (out.probs[a] === 0) a--;  // guard against rounding landing on an illegal action
    return { action: a, probs: out.probs };
  };

  var api = {
    FOLD: FOLD, CALL: CALL, HALF_POT: HALF_POT, POT: POT, ALL_IN: ALL_IN, CATEGORIES: CATEGORIES, MAX_PLAYERS: MAX_PLAYERS,
    evaluate: evaluate, equityMC: equityMC, preflopClass: preflopClass, Env: Env, encode: encode, encodeHU: encodeHU,
    Policy: Policy, Bot: Bot,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.PokerEngine = api;
})(this);
