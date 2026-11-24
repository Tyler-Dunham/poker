# Poker simulator

A small, dependency-free, no-limit Texas Hold'em simulator for Python 3.9+ bots. It supports 2–22 seats, blinds, betting across all streets, all-ins, side pots, and split pots. Bots receive only their own hole cards and public game state.

Run the example from this directory:

```sh
python3 -m examples.play
```

Create a bot by inheriting from `Bot` and implementing `act(observation) -> Action`:

```python
from poker import Action, Bot, PokerGame

class MyBot(Bot):
    def act(self, observation):
        if observation.to_call:
            return Action("call")
        return Action("check")

game = PokerGame([MyBot(), MyBot()], seed=42)
result = game.play_hand()
print(result.net, result.stacks)
```

`observation.legal_actions` lists valid action names. For `bet` and `raise`, pass `Action(kind, amount)` where `amount` is the **total committed on the current street**. The allowed range is `observation.min_raise_to` through `observation.max_raise_to`; a short all-in has the same minimum and maximum. `observation.to_call` is the additional amount needed to call. A `call` automatically uses the player's remaining stack if it is short.

`observation` also includes the board, pot, stacks, current street bets, folded seats, button, and public action history. `HandResult` includes per-seat contributions, payouts, net chip change, final stacks, board, action events, and showdown hands. This gives data-collection and learning bots a stable input and a per-hand reward (`result.net[seat]`). Seats and bots stay in the same order across hands; the button moves among players with chips. `game.run(hands)` plays up to that many hands, stopping if only one player has chips.

Run tests with `python3 -m unittest discover -s tests`.

## Poker Mirror & RL Exploiter Web App

Start the dependency-free web studio (`ThreadingHTTPServer` + vanilla HTML/CSS/JS):

```sh
python3 -m web.server --port 8081
```

Then open `http://127.0.0.1:8081` to import hand histories or load sample player packs, inspect your 13×13 preflop range matrix, train an RL Exploiter against your behavioral clone, and play interactive hands with live GTO vs. Clone coaching.

Run the web API integration tests:

```sh
python3 -m unittest tests/test_web_api.py
```
