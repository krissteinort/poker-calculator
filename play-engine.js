// Heads-up engine for play.html: a JS mirror of bot/poker (cards.py, env.py, features.py, agents.forward).
// Keep it in sync with the Python; bot/tests/test_parity.py checks that the two agree.
(function (root) {
  var FOLD = 0, CALL = 1, HALF_POT = 2, POT = 3, ALL_IN = 4;
  var BOARD_COUNT = [0, 3, 4, 5, 5];
  var OBS_SIZE = 140;
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

  function equityMC(hole, board, samples) {
    var known = {}, deck = [], i, j;
    hole.concat(board).forEach(function (c) { known[c] = true; });
    for (i = 0; i < 52; i++) if (!known[i]) deck.push(i);
    var need = 5 - board.length, total = 0;
    for (var s = 0; s < samples; s++) {
      for (i = 0; i < need + 2; i++) {
        j = i + Math.floor(Math.random() * (deck.length - i));
        var t = deck[i]; deck[i] = deck[j]; deck[j] = t;
      }
      var b = board.concat(deck.slice(2, 2 + need));
      var me = evaluate(hole.concat(b)), op = evaluate([deck[0], deck[1]].concat(b));
      total += me > op ? 1 : me === op ? 0.5 : 0;
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
  function Env(stackBB, bb) {
    this.bb = bb || 2;
    this.startStack = (stackBB || 100) * this.bb;
  }
  // stacks: starting chips per seat (default startStack each); each must cover a big blind.
  Env.prototype.reset = function (deck, button, stacks) {
    this.hole = [deck.slice(0, 2), deck.slice(2, 4)];
    this.fullBoard = deck.slice(4, 9);
    this.button = button;
    this.stack = stacks ? stacks.slice() : [this.startStack, this.startStack];
    this.bet = [0, 0];
    this.contrib = [0, 0];
    this.street = 0;
    this.history = [[], [], [], []];
    this.done = false;
    this.payoff = null;
    this._put(button, this.bb / 2);
    this._put(1 - button, this.bb);
    this.lastRaise = this.bb;
    this.acted = [false, false];
    this.toAct = button;
  };
  Env.prototype.board = function () { return this.fullBoard.slice(0, BOARD_COUNT[this.street]); };
  Env.prototype.pot = function () { return this.contrib[0] + this.contrib[1]; };
  Env.prototype.toCall = function (seat) { return this.bet[1 - seat] - this.bet[seat]; };
  Env.prototype._put = function (seat, amount) {
    amount = Math.min(amount, this.stack[seat]);
    this.stack[seat] -= amount;
    this.bet[seat] += amount;
    this.contrib[seat] += amount;
  };
  Env.prototype.raiseAmount = function (seat, frac) {
    var tc = this.toCall(seat);
    return tc + Math.max(Math.floor(frac * (this.pot() + tc)), this.lastRaise, this.bb);
  };
  Env.prototype.legalActions = function () {
    var seat = this.toAct, tc = this.toCall(seat), acts = tc > 0 ? [FOLD] : [];
    acts.push(CALL);
    if (this.stack[seat] > tc && this.stack[1 - seat] > 0) {
      if (this.raiseAmount(seat, 0.5) < this.stack[seat]) acts.push(HALF_POT);
      if (this.raiseAmount(seat, 1) < this.stack[seat]) acts.push(POT);
      acts.push(ALL_IN);
    }
    return acts;
  };
  // Chips `seat` would add for action a (for labels).
  Env.prototype.amountFor = function (a) {
    var seat = this.toAct;
    if (a === FOLD) return 0;
    if (a === CALL) return Math.min(this.toCall(seat), this.stack[seat]);
    if (a === ALL_IN) return this.stack[seat];
    return this.raiseAmount(seat, a === HALF_POT ? 0.5 : 1);
  };
  Env.prototype.step = function (a) {
    var seat = this.toAct, opp = 1 - seat;
    if (this.done || this.legalActions().indexOf(a) < 0) throw new Error('illegal action ' + a);
    this.history[this.street].push([seat, a]);
    if (a === FOLD) {
      var lost = this.contrib[seat], net = [0, 0];
      net[seat] = -lost; net[opp] = lost;
      return this._finish(net);
    }
    if (a === CALL) {
      this._put(seat, this.toCall(seat));
      this.acted[seat] = true;
    } else {
      var amount = a === ALL_IN ? this.stack[seat] : this.raiseAmount(seat, a === HALF_POT ? 0.5 : 1);
      var raiseBy = this.bet[seat] + amount - this.bet[opp];
      this._put(seat, amount);
      this.lastRaise = Math.max(this.lastRaise, raiseBy);
      this.acted = [false, false];
      this.acted[seat] = true;
    }
    var matched = this.bet[0] === this.bet[1] || (Math.min(this.stack[0], this.stack[1]) === 0 && this.bet[seat] <= this.bet[opp]);
    if (this.acted[0] && this.acted[1] && matched) this._nextStreet();
    else this.toAct = opp;
  };
  Env.prototype._nextStreet = function () {
    if (Math.min(this.stack[0], this.stack[1]) === 0 || this.street === 3) {
      this.street = 4;
      return this._showdown();
    }
    this.street++;
    this.bet = [0, 0];
    this.acted = [false, false];
    this.lastRaise = this.bb;
    this.toAct = 1 - this.button;
  };
  Env.prototype._showdown = function () {
    var m = Math.min(this.contrib[0], this.contrib[1]);
    var s0 = evaluate(this.hole[0].concat(this.fullBoard)), s1 = evaluate(this.hole[1].concat(this.fullBoard));
    if (s0 === s1) return this._finish([0, 0]);
    this._finish(s0 > s1 ? [m, -m] : [-m, m]);
  };
  Env.prototype._finish = function (net) {
    this.done = true;
    this.payoff = [net[0] / this.bb, net[1] / this.bb];
  };

  // ---- features (mirrors features.py) ----
  function encode(env, seat, eq) {
    var x = new Float32Array(OBS_SIZE), board = env.board(), i;
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
    var logits = this.pi.apply(h, false), max = -Infinity, probs = [], sum = 0, a;
    for (a = 0; a < logits.length; a++) {
      if (legal.indexOf(a) < 0) logits[a] = -1e9;
      if (logits[a] > max) max = logits[a];
    }
    for (a = 0; a < logits.length; a++) { probs.push(Math.exp(logits[a] - max)); sum += probs[a]; }
    return { probs: probs.map(function (p) { return p / sum; }), value: this.v.apply(h, false)[0] };
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

  // The bot as an opponent: caches equity per street like EquityCache in features.py.
  function Bot(json) {
    this.policy = new Policy(decodeModel(json));
    this.preflop = json.preflop;
    this.cache = {};
  }
  Bot.prototype.newHand = function () { this.cache = {}; };
  Bot.prototype.equity = function (env, seat) {
    var key = seat + ':' + env.street;
    if (!(key in this.cache)) {
      var board = env.board();
      this.cache[key] = board.length ? equityMC(env.hole[seat], board, 128) : this.preflop[preflopClass(env.hole[seat])];
    }
    return this.cache[key];
  };
  Bot.prototype.act = function (env, seat) {
    var out = this.policy.forward(encode(env, seat, this.equity(env, seat)), env.legalActions());
    var r = Math.random(), a = 0;
    for (; a < out.probs.length - 1; a++) { r -= out.probs[a]; if (r < 0) break; }
    while (out.probs[a] === 0) a--;  // guard against rounding landing on an illegal action
    return { action: a, probs: out.probs };
  };

  var api = {
    FOLD: FOLD, CALL: CALL, HALF_POT: HALF_POT, POT: POT, ALL_IN: ALL_IN, CATEGORIES: CATEGORIES,
    evaluate: evaluate, equityMC: equityMC, preflopClass: preflopClass, Env: Env, encode: encode,
    Policy: Policy, Bot: Bot,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.PokerEngine = api;
})(this);
