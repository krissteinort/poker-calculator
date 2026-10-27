import random
import unittest

from poker.agents import BASELINES
from poker.cards import cards, category, equity_mc, evaluate
from poker.env import ALL_IN, CALL, FOLD, POT, HeadsUpEnv
from poker.features import OBS_SIZE, EquityCache, encode
from poker.play import play_hand


class TestCards(unittest.TestCase):
    def test_categories(self):
        cases = {
            '2s 3s 4s 5s 6s 9d Kc': 8, 'As Ks Qs Js Ts 2d 3c': 8, 'As 2d 3h 4c 5s 9d Kc': 4,
            'Ah Ad As Kh Kd 2c 3c': 6, '9s 9h 9d 9c 2c 3c 4d': 7, '2h 5h 9h Jh Kh 3c 4d': 5,
            'Ah Ad As 2h 5d 7c 9c': 3, 'Ah Ad Kh Kd 2c 2d 9c': 2, 'Ah Ad 3h 5d 7c 9c Jd': 1,
            'Ah 2d 4h 7d 9c Jc Kd': 0, 'Ah Ad Ac Kh Kd Kc 2c': 6,
        }
        for hand, cat in cases.items():
            self.assertEqual(category(cards(hand)), cat, hand)

    def test_ordering(self):
        wheel = evaluate(cards('As 2d 3h 4c 5s 9d Kc'))
        six_high = evaluate(cards('2s 3d 4h 5c 6s 9d Kc'))
        self.assertLess(wheel, six_high)
        self.assertGreater(evaluate(cards('Ah Ad Kc Qd 2s 3h 7c')), evaluate(cards('Ah Ad Kc Jd 2s 3h 7c')))

    def test_equity(self):
        self.assertAlmostEqual(equity_mc(cards('Ah Ad'), [], 20000, random.Random(1)), 0.852, delta=0.01)


class TestEnv(unittest.TestCase):
    def test_random_play_conserves_chips(self):
        rng = random.Random(0)
        env = HeadsUpEnv()
        agents = [BASELINES['random'](), BASELINES['random']()]
        for i in range(3000):
            payoff = play_hand(env, agents, None, i % 2, rng)
            self.assertAlmostEqual(payoff[0] + payoff[1], 0)
            self.assertLessEqual(abs(payoff[0]), 100)
            self.assertEqual(env.contrib[0] + env.stack[0], env.start_stack)

    def test_fold_preflop(self):
        env = HeadsUpEnv()
        env.reset(list(range(52)), button=0)
        self.assertEqual(env.legal_actions()[0], FOLD)
        env.step(FOLD)
        self.assertEqual(env.payoff, [-0.5, 0.5])

    def test_limp_and_check_down(self):
        env = HeadsUpEnv()
        env.reset(cards('Ah Kd 7c 2s') + [c for c in range(52) if c not in cards('Ah Kd 7c 2s')], button=1)
        env.step(CALL)  # button limps
        self.assertEqual(env.to_act, 0)  # big blind has the option
        self.assertNotIn(FOLD, env.legal_actions())
        env.step(CALL)
        self.assertEqual(env.street, 1)
        self.assertEqual(env.to_act, 0)  # big blind acts first after the flop
        while not env.done:
            env.step(CALL)
        self.assertEqual(abs(env.payoff[0]), 1)

    def test_all_in_called(self):
        env = HeadsUpEnv()
        env.reset(cards('Ah Ad Kh Kd 2c 7s 9d Jc 3s') + [c for c in range(52) if c not in cards('Ah Ad Kh Kd 2c 7s 9d Jc 3s')], button=0)
        env.step(ALL_IN)
        env.step(CALL)
        self.assertTrue(env.done)
        self.assertEqual(env.payoff, [100, -100])

    def test_pot_raise_size(self):
        env = HeadsUpEnv()
        env.reset(None, button=0)
        env.step(POT)  # SB 1 calls 1 more (pot 4) then raises 4: puts in 5, total bet 6 = 3bb
        self.assertEqual(env.bet[0], 6)

    def test_features(self):
        rng = random.Random(0)
        env = HeadsUpEnv()
        env.reset(None, 0, rng)
        x = encode(env, 0, EquityCache(rng=rng).get(env, 0))
        self.assertEqual(x.shape, (OBS_SIZE,))
        self.assertEqual(x[:52].sum(), 2)


if __name__ == '__main__':
    unittest.main()
