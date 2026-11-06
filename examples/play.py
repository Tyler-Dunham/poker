from random import Random

from poker import Action, Bot, Observation, PokerGame


class CallingBot(Bot):
    def act(self, observation: Observation) -> Action:
        return Action("call" if observation.to_call else "check")


class RandomBot(Bot):
    def __init__(self, seed: int):
        self.random = Random(seed)

    def act(self, observation: Observation) -> Action:
        choice = self.random.choice(observation.legal_actions)
        amount = observation.min_raise_to if choice in ("bet", "raise") else 0
        return Action(choice, amount or 0)


if __name__ == "__main__":
    game = PokerGame([CallingBot(), RandomBot(1), CallingBot()], seed=42)
    for number, result in enumerate(game.run(10), 1):
        print(f"Hand {number}: board={result.board} net={result.net} stacks={result.stacks}")
