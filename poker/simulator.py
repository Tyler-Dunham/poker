"""Small no-limit Texas Hold'em simulator. Chip amounts are integers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from random import Random


RANKS = "23456789TJQKA"
SUITS = "cdhs"


def hand_rank(cards: tuple[str, ...]) -> tuple[int, ...]:
    """Rank five to seven cards; larger tuples win."""
    def five(hand: tuple[str, ...]) -> tuple[int, ...]:
        values = [RANKS.index(card[0]) + 2 for card in hand]
        counts = Counter(values)
        groups = sorted(((count, value) for value, count in counts.items()), reverse=True)
        unique = set(values)
        if 14 in unique:
            unique.add(1)
        straight = max((high for high in range(5, 15) if set(range(high - 4, high + 1)) <= unique), default=0)
        flush = len({card[1] for card in hand}) == 1
        if straight and flush:
            return (8, straight)
        if groups[0][0] == 4:
            return (7, groups[0][1], groups[1][1])
        if groups[0][0] == 3 and groups[1][0] == 2:
            return (6, groups[0][1], groups[1][1])
        if flush:
            return (5, *sorted(values, reverse=True))
        if straight:
            return (4, straight)
        if groups[0][0] == 3:
            return (3, groups[0][1], *sorted((v for v in values if v != groups[0][1]), reverse=True))
        if groups[0][0] == groups[1][0] == 2:
            pairs = sorted((value for count, value in groups if count == 2), reverse=True)
            return (2, *pairs, next(v for v in values if v not in pairs))
        if groups[0][0] == 2:
            return (1, groups[0][1], *sorted((v for v in values if v != groups[0][1]), reverse=True))
        return (0, *sorted(values, reverse=True))

    if not 5 <= len(cards) <= 7:
        raise ValueError("hand_rank needs five to seven cards")
    return max(five(hand) for hand in combinations(cards, 5))


@dataclass(frozen=True)
class Action:
    kind: str  # fold, check, call, bet, raise
    amount: int = 0  # for bet/raise: total chips committed on this street


@dataclass(frozen=True)
class ActionEvent:
    street: str
    seat: int
    kind: str
    amount: int  # chips added by this action
    pot: int  # pot after this action


@dataclass(frozen=True)
class Observation:
    seat: int
    button: int
    street: str
    hole: tuple[str, str]
    board: tuple[str, ...]
    pot: int
    stacks: tuple[int, ...]
    bets: tuple[int, ...]
    folded: tuple[bool, ...]
    to_call: int
    min_raise_to: int | None
    max_raise_to: int | None
    legal_actions: tuple[str, ...]
    history: tuple[ActionEvent, ...]


class Bot(ABC):
    @abstractmethod
    def act(self, observation: Observation) -> Action:
        """Choose an action from the current observation."""


@dataclass(frozen=True)
class HandResult:
    board: tuple[str, ...]
    contributions: tuple[int, ...]
    payouts: tuple[int, ...]
    net: tuple[int, ...]
    stacks: tuple[int, ...]
    events: tuple[ActionEvent, ...]
    showdown: dict[int, tuple[str, str]]


def _award_pots(contributions: list[int], folded: set[int], ranks: dict[int, tuple[int, ...]], button: int) -> list[int]:
    payouts = [0] * len(contributions)
    previous = 0
    for level in sorted(set(contributions) - {0}):
        participants = [seat for seat, amount in enumerate(contributions) if amount >= level]
        pot = (level - previous) * len(participants)
        eligible = [seat for seat in participants if seat not in folded]
        best = max(ranks[seat] for seat in eligible)
        winners = [seat for seat in eligible if ranks[seat] == best]
        share, extra = divmod(pot, len(winners))
        for seat in winners:
            payouts[seat] += share
        for seat in sorted(winners, key=lambda seat: (seat - button - 1) % len(contributions))[:extra]:
            payouts[seat] += 1
        previous = level
    return payouts


class PokerGame:
    def __init__(self, bots: list[Bot], starting_stack: int = 1000, small_blind: int = 5,
                 big_blind: int = 10, seed: int | None = None):
        if not 2 <= len(bots) <= 22 or not 0 < small_blind < big_blind or starting_stack < big_blind:
            raise ValueError("need 2-22 bots, valid blinds, and a starting stack of at least one big blind")
        if any(not isinstance(bot, Bot) for bot in bots):
            raise TypeError("every player must inherit from Bot")
        self.bots = bots
        self.stacks = [starting_stack] * len(bots)
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.random = Random(seed)
        self.button = -1

    def play_hand(self) -> HandResult:
        live = [seat for seat, stack in enumerate(self.stacks) if stack]
        if len(live) < 2:
            raise ValueError("fewer than two players have chips")
        n = len(self.bots)

        def next_live(seat: int) -> int:
            return next((seat + step) % n for step in range(1, n + 1) if (seat + step) % n in live)

        self.button = next_live(self.button)
        small = self.button if len(live) == 2 else next_live(self.button)
        big = next_live(small)
        deck = [rank + suit for rank in RANKS for suit in SUITS]
        self.random.shuffle(deck)
        holes = {seat: (deck.pop(), deck.pop()) for seat in live}
        board: list[str] = []
        folded: set[int] = set()
        contributions = [0] * n
        bets = [0] * n
        events: list[ActionEvent] = []

        def pay(seat: int, amount: int, street: str, kind: str) -> None:
            amount = min(amount, self.stacks[seat])
            self.stacks[seat] -= amount
            contributions[seat] += amount
            bets[seat] += amount
            events.append(ActionEvent(street, seat, kind, amount, sum(contributions)))

        pay(small, self.small_blind, "preflop", "small_blind")
        pay(big, self.big_blind, "preflop", "big_blind")

        def bet_round(street: str, start: int, current_bet: int) -> None:
            last_full_raise = self.big_blind
            acted: set[int] = set()
            pending = {seat for seat in live if seat not in folded and self.stacks[seat]}
            order = [(start + step) % n for step in range(n)]
            turn = 0
            while pending and len(live) - len(folded) > 1:
                seat = order[turn % n]
                turn += 1
                if seat not in pending:
                    continue
                to_call = max(0, current_bet - bets[seat])
                max_to = bets[seat] + self.stacks[seat]
                can_raise = (seat not in acted and max_to > current_bet and
                             any(other != seat and other not in folded and self.stacks[other] for other in live))
                min_to = max(self.big_blind, current_bet + last_full_raise)
                if not can_raise:
                    min_to = max_to = None
                elif max_to < min_to:
                    min_to = max_to  # short all-in raise
                raise_kind = "bet" if current_bet == 0 else "raise"
                legal = ["fold", "call"] if to_call else ["check"]
                if can_raise:
                    legal.append(raise_kind)
                observation = Observation(seat, self.button, street, holes[seat], tuple(board),
                                          sum(contributions), tuple(self.stacks), tuple(bets),
                                          tuple(i in folded for i in range(n)), to_call,
                                          min_to, max_to, tuple(legal), tuple(events))
                action = self.bots[seat].act(observation)
                if action.kind not in legal:
                    raise ValueError(f"seat {seat}: illegal action {action.kind}")
                pending.remove(seat)
                if action.kind == "fold":
                    folded.add(seat)
                    events.append(ActionEvent(street, seat, "fold", 0, sum(contributions)))
                elif action.kind == "check":
                    events.append(ActionEvent(street, seat, "check", 0, sum(contributions)))
                elif action.kind == "call":
                    pay(seat, to_call, street, "call")
                else:
                    if not isinstance(action.amount, int) or not min_to <= action.amount <= max_to:
                        raise ValueError(f"seat {seat}: invalid {action.kind} amount {action.amount}")
                    old_bet = current_bet
                    pay(seat, action.amount - bets[seat], street, action.kind)
                    current_bet = bets[seat]
                    full_raise = current_bet >= self.big_blind if old_bet < self.big_blind else current_bet - old_bet >= last_full_raise
                    if full_raise:
                        last_full_raise = current_bet if old_bet < self.big_blind else current_bet - old_bet
                        acted.clear()
                        pending = {other for other in live if other != seat and other not in folded and self.stacks[other]}
                    else:
                        pending = {other for other in live if other != seat and other not in folded and
                                   self.stacks[other] and bets[other] < current_bet}
                acted.add(seat)

        bet_round("preflop", next_live(big), self.big_blind)
        for street, cards in (("flop", 3), ("turn", 1), ("river", 1)):
            if len(live) - len(folded) == 1:
                break
            deck.pop()  # burn
            board.extend(deck.pop() for _ in range(cards))
            bets = [0] * n
            bet_round(street, next_live(self.button), 0)

        contenders = [seat for seat in live if seat not in folded]
        ranks = {seat: hand_rank((*holes[seat], *board)) for seat in contenders} if len(contenders) > 1 else {contenders[0]: (0,)}
        payouts = _award_pots(contributions, folded, ranks, self.button)
        for seat, amount in enumerate(payouts):
            self.stacks[seat] += amount
        return HandResult(tuple(board), tuple(contributions), tuple(payouts),
                          tuple(payouts[i] - contributions[i] for i in range(n)),
                          tuple(self.stacks), tuple(events),
                          {seat: holes[seat] for seat in contenders} if len(contenders) > 1 else {})

    def run(self, hands: int) -> list[HandResult]:
        results = []
        for _ in range(hands):
            if sum(stack > 0 for stack in self.stacks) < 2:
                break
            results.append(self.play_hand())
        return results
