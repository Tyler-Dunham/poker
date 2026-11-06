import unittest

from poker import Action, Bot, PokerGame, hand_rank
from poker.simulator import _award_pots


class CallingBot(Bot):
    def __init__(self):
        self.observations = []

    def act(self, observation):
        self.observations.append(observation)
        return Action("call" if observation.to_call else "check")


class AllInBot(Bot):
    def act(self, observation):
        if "raise" in observation.legal_actions:
            return Action("raise", observation.max_raise_to)
        if "bet" in observation.legal_actions:
            return Action("bet", observation.max_raise_to)
        return Action("call" if observation.to_call else "check")


class SimulatorTests(unittest.TestCase):
    def test_bot_must_implement_act(self):
        class IncompleteBot(Bot):
            pass

        with self.assertRaises(TypeError):
            IncompleteBot()
        with self.assertRaises(TypeError):
            PokerGame([object(), CallingBot()])

    def test_hand_rank(self):
        self.assertGreater(hand_rank(("As", "2s", "3s", "4s", "5s")),
                           hand_rank(("Ac", "Ad", "Ah", "As", "2d")))
        self.assertGreater(hand_rank(("9c", "9d", "9h", "2s", "2d")),
                           hand_rank(("Ac", "Jc", "8c", "4c", "2c")))
        self.assertEqual(hand_rank(("As", "2d", "3c", "4h", "5s")), (4, 5))

    def test_side_pot(self):
        payouts = _award_pots([100, 50, 100], set(), {0: (1,), 1: (3,), 2: (2,)}, 0)
        self.assertEqual(payouts, [0, 150, 100])

    def test_bots_play_and_chips_are_conserved(self):
        bots = [CallingBot(), AllInBot(), CallingBot()]
        game = PokerGame(bots, starting_stack=100, seed=8)
        results = game.run(20)
        self.assertTrue(results)
        self.assertEqual(sum(game.stacks), 300)
        self.assertTrue(all(sum(result.net) == 0 for result in results))
        self.assertTrue(all(len(result.board) in (0, 3, 4, 5) for result in results))
        self.assertTrue(bots[0].observations)
        self.assertFalse(hasattr(bots[0].observations[0], "other_holes"))


if __name__ == "__main__":
    unittest.main()
