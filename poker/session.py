"""Interactive HU and 6-max hand session with Live Mirror Coach and JSON export."""

from __future__ import annotations

import json
from random import Random

from .abstraction import (
    ACTION_SPACE,
    abstract_to_legal_action,
    encode_observation,
    legal_abstract_actions,
    position_name,
)
from .player_model import PlayerCloneBot, PlayerCloneModel
from .rl_exploiter import GTOBaselineBot, RLExploiterBot
from .simulator import (
    RANKS,
    SUITS,
    Action,
    ActionEvent,
    Bot,
    HandResult,
    Observation,
    _award_pots,
    hand_rank,
)


class InteractiveHandSession:
    """Step-by-step interactive poker table pausing on Seat 0's turn."""

    def __init__(
        self,
        mode: str = "self_clone",
        num_players: int = 2,
        starting_stack: int = 1000,
        small_blind: int = 5,
        big_blind: int = 10,
        hero_name: str = "Hero",
        clone_model: PlayerCloneModel | None = None,
        exploiter_bot: RLExploiterBot | None = None,
        gto_bot: GTOBaselineBot | None = None,
        seed: int | None = 42,
    ) -> None:
        if num_players not in (2, 6):
            raise ValueError("num_players must be 2 (HU) or 6 (6-max)")
        if mode not in ("self_clone", "rl_exploiter", "gto", "mixed_table"):
            raise ValueError(f"Unsupported session mode: {mode!r}")

        self.mode = mode
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.hero_name = hero_name
        self.rng = Random(seed)

        self.clone_model = clone_model or PlayerCloneModel()
        self.exploiter_bot = exploiter_bot or RLExploiterBot(seed=seed)
        self.gto_bot = gto_bot or GTOBaselineBot(seed=seed)

        self.bots: dict[int, Bot] = {}
        self.seat_names: list[str] = [hero_name]
        for s in range(1, num_players):
            if mode == "self_clone":
                self.bots[s] = PlayerCloneBot(self.clone_model, seed=(seed or 0) + s)
                self.seat_names.append(f"Clone_{s}")
            elif mode == "rl_exploiter":
                self.bots[s] = self.exploiter_bot
                self.seat_names.append(f"Exploiter_{s}")
            elif mode == "gto":
                self.bots[s] = self.gto_bot
                self.seat_names.append(f"GTO_{s}")
            else:  # mixed_table
                if s % 3 == 1:
                    self.bots[s] = PlayerCloneBot(self.clone_model, seed=(seed or 0) + s)
                    self.seat_names.append(f"Clone_{s}")
                elif s % 3 == 2:
                    self.bots[s] = self.exploiter_bot
                    self.seat_names.append(f"Exploiter_{s}")
                else:
                    self.bots[s] = self.gto_bot
                    self.seat_names.append(f"GTO_{s}")

        self.stacks = [starting_stack] * num_players
        self.button = -1
        self.hand_counter = 0

        # Active hand state
        self.hand_id = ""
        self.live: list[int] = []
        self.deck: list[str] = []
        self.holes: dict[int, tuple[str, str]] = {}
        self.board: list[str] = []
        self.folded: set[int] = set()
        self.contributions: list[int] = []
        self.bets: list[int] = []
        self.events: list[ActionEvent] = []
        self.street = "preflop"
        self.current_bet = 0
        self.last_full_raise = big_blind
        self.acted: set[int] = set()
        self.pending: set[int] = set()
        self.order: list[int] = []
        self.turn = 0
        self.awaiting_hero = False
        self.hand_over = True
        self.current_obs: Observation | None = None
        self.last_result: HandResult | None = None
        self.coach_feedback: list[dict] = []
        self.hero_decisions_log: list[dict] = []
        self.completed_hands: list[dict] = []

        # Trackers for stats flags
        self._preflop_raises = 0
        self._preflop_aggressor: int | None = None
        self._hero_opened_pf = False
        self._hero_vpip_done = False
        self._hero_pfr_done = False
        self._cbet_made = False
        self._cbet_bettor: int | None = None

    def _next_live(self, seat: int) -> int:
        n = self.num_players
        return next((seat + step) % n for step in range(1, n + 1) if (seat + step) % n in self.live)

    def _pay(self, seat: int, amount: int, street: str, kind: str) -> int:
        amount = min(amount, self.stacks[seat])
        self.stacks[seat] -= amount
        self.contributions[seat] += amount
        self.bets[seat] += amount
        self.events.append(ActionEvent(street, seat, kind, amount, sum(self.contributions)))
        return amount

    def start_hand(self, reset_busted: bool = True) -> dict:
        """Deal a new hand and advance until Seat 0's turn or hand completion."""
        if reset_busted:
            for i in range(self.num_players):
                if self.stacks[i] < self.big_blind:
                    self.stacks[i] = self.starting_stack

        self.live = [s for s, st in enumerate(self.stacks) if st > 0]
        if len(self.live) < 2:
            raise ValueError("Need at least two players with chips")

        self.hand_counter += 1
        self.hand_id = f"session_{self.hand_counter}"
        self.button = self._next_live(self.button)
        small = self.button if len(self.live) == 2 else self._next_live(self.button)
        big = self._next_live(small)

        self.deck = [r + s for r in RANKS for s in SUITS]
        self.rng.shuffle(self.deck)
        self.holes = {seat: (self.deck.pop(), self.deck.pop()) for seat in self.live}
        self.board = []
        self.folded = set()
        self.contributions = [0] * self.num_players
        self.bets = [0] * self.num_players
        self.events = []
        self.coach_feedback = []
        self.hero_decisions_log = []
        self.last_result = None
        self.hand_over = False
        self.awaiting_hero = False
        self.current_obs = None

        self._preflop_raises = 0
        self._preflop_aggressor = None
        self._hero_opened_pf = False
        self._hero_vpip_done = False
        self._hero_pfr_done = False
        self._cbet_made = False
        self._cbet_bettor = None

        self._pay(small, self.small_blind, "preflop", "small_blind")
        self._pay(big, self.big_blind, "preflop", "big_blind")

        self._setup_bet_round("preflop", self._next_live(big), self.big_blind)
        self._advance()
        return self.get_state()

    def _setup_bet_round(self, street: str, start: int, current_bet: int) -> None:
        self.street = street
        self.current_bet = current_bet
        self.last_full_raise = self.big_blind
        self.acted = set()
        self.pending = {s for s in self.live if s not in self.folded and self.stacks[s] > 0}
        self.order = [(start + step) % self.num_players for step in range(self.num_players)]
        self.turn = 0

    def _build_observation(self, seat: int) -> Observation:
        to_call = max(0, self.current_bet - self.bets[seat])
        max_to = self.bets[seat] + self.stacks[seat]
        can_raise = (
            seat not in self.acted
            and max_to > self.current_bet
            and any(other != seat and other not in self.folded and self.stacks[other] > 0 for other in self.live)
        )
        min_to = max(self.big_blind, self.current_bet + self.last_full_raise)
        if not can_raise:
            min_to = max_to = None
        elif max_to < min_to:
            min_to = max_to

        raise_kind = "bet" if self.current_bet == 0 else "raise"
        legal = ["fold", "call"] if to_call else ["check"]
        if can_raise:
            legal.append(raise_kind)

        return Observation(
            seat=seat,
            button=self.button,
            street=self.street,
            hole=self.holes[seat],
            board=tuple(self.board),
            pot=sum(self.contributions),
            stacks=tuple(self.stacks),
            bets=tuple(self.bets),
            folded=tuple(i in self.folded for i in range(self.num_players)),
            to_call=to_call,
            min_raise_to=min_to,
            max_raise_to=max_to,
            legal_actions=tuple(legal),
            history=tuple(self.events),
        )

    def _apply_action(self, seat: int, action: Action, obs: Observation) -> None:
        if action.kind not in obs.legal_actions:
            raise ValueError(f"seat {seat}: illegal action {action.kind}")
        self.pending.discard(seat)
        added = 0

        if action.kind == "fold":
            self.folded.add(seat)
            self.events.append(ActionEvent(self.street, seat, "fold", 0, sum(self.contributions)))
        elif action.kind == "check":
            self.events.append(ActionEvent(self.street, seat, "check", 0, sum(self.contributions)))
        elif action.kind == "call":
            added = self._pay(seat, obs.to_call, self.street, "call")
        else:
            if not isinstance(action.amount, int) or obs.min_raise_to is None or obs.max_raise_to is None:
                raise ValueError(f"seat {seat}: invalid {action.kind}")
            if not obs.min_raise_to <= action.amount <= obs.max_raise_to:
                raise ValueError(f"seat {seat}: invalid {action.kind} amount {action.amount}")
            old_bet = self.current_bet
            added = self._pay(seat, action.amount - self.bets[seat], self.street, action.kind)
            self.current_bet = self.bets[seat]
            full_raise = (
                self.current_bet >= self.big_blind
                if old_bet < self.big_blind
                else self.current_bet - old_bet >= self.last_full_raise
            )
            if full_raise:
                self.last_full_raise = self.current_bet if old_bet < self.big_blind else self.current_bet - old_bet
                self.acted.clear()
                self.pending = {
                    other for other in self.live
                    if other != seat and other not in self.folded and self.stacks[other] > 0
                }
            else:
                self.pending = {
                    other for other in self.live
                    if other != seat and other not in self.folded
                    and self.stacks[other] > 0 and self.bets[other] < self.current_bet
                }

        self.acted.add(seat)

        if self.street == "preflop" and action.kind in ("bet", "raise"):
            self._preflop_raises += 1
            self._preflop_aggressor = seat
            if self._preflop_raises == 1 and seat == 0:
                self._hero_opened_pf = True
        elif self.street == "flop" and action.kind == "bet" and seat == self._preflop_aggressor:
            self._cbet_made = True
            self._cbet_bettor = seat

    def _finish_hand(self) -> None:
        contenders = [s for s in self.live if s not in self.folded]
        ranks = (
            {s: hand_rank((*self.holes[s], *self.board)) for s in contenders}
            if len(contenders) > 1
            else {contenders[0]: (0,)}
        )
        payouts = _award_pots(self.contributions, self.folded, ranks, self.button)
        for s, amt in enumerate(payouts):
            self.stacks[s] += amt

        net = tuple(payouts[i] - self.contributions[i] for i in range(self.num_players))
        showdown = {s: self.holes[s] for s in contenders} if len(contenders) > 1 else {}
        self.last_result = HandResult(
            board=tuple(self.board),
            contributions=tuple(self.contributions),
            payouts=tuple(payouts),
            net=net,
            stacks=tuple(self.stacks),
            events=tuple(self.events),
            showdown=showdown,
        )
        self.hand_over = True
        self.awaiting_hero = False
        self.current_obs = None

        went_sd = len(contenders) > 1 and 0 in contenders
        won_sd = went_sd and payouts[0] > 0
        hero_pos = position_name(0, self.button, self.num_players)
        self.completed_hands.append({
            "hand_id": self.hand_id,
            "hero_name": self.hero_name,
            "hero_hole": list(self.holes[0]) if 0 in self.holes else ["As", "Kd"],
            "position": hero_pos,
            "board": list(self.board),
            "big_blind": self.big_blind,
            "net_chips": net[0],
            "net_bb": round(net[0] / max(1, self.big_blind), 2),
            "went_to_showdown": went_sd,
            "won_at_showdown": won_sd,
            "decisions": list(self.hero_decisions_log),
        })

    def _advance(self) -> None:
        street_cards = {"preflop": ("flop", 3), "flop": ("turn", 1), "turn": ("river", 1)}

        while not self.hand_over:
            while self.pending and len(self.live) - len(self.folded) > 1:
                seat = self.order[self.turn % self.num_players]
                self.turn += 1
                if seat not in self.pending:
                    continue

                obs = self._build_observation(seat)
                if seat == 0:
                    self.awaiting_hero = True
                    self.current_obs = obs
                    return

                action = self.bots[seat].act(obs)
                self._apply_action(seat, action, obs)

            if len(self.live) - len(self.folded) <= 1 or self.street == "river":
                self._finish_hand()
                return

            next_street, num_cards = street_cards[self.street]
            self.deck.pop()  # burn
            self.board.extend(self.deck.pop() for _ in range(num_cards))
            self.bets = [0] * self.num_players
            self._setup_bet_round(next_street, self._next_live(self.button), 0)

    def submit_action(self, action: str | Action) -> dict:
        """Submit Seat 0's action, record Live Mirror Coach feedback, and advance."""
        if not self.awaiting_hero or self.current_obs is None:
            raise RuntimeError("Session is not awaiting Seat 0's action")

        obs = self.current_obs
        if isinstance(action, Action):
            sim_action = action
            if action.kind == "fold":
                abstract_act = "fold"
            elif action.kind in ("check", "call"):
                abstract_act = "check_call"
            else:
                pot_after = max(1, obs.pot + obs.to_call)
                ratio = max(0, action.amount - obs.bets[0] - obs.to_call) / pot_after
                abstract_act = "bet_small" if ratio <= 0.48 else ("bet_medium" if ratio <= 0.90 else "bet_large")
        else:
            if action not in ACTION_SPACE:
                raise ValueError(f"Unknown abstract action: {action!r}")
            abstract_act = action
            sim_action = abstract_to_legal_action(abstract_act, obs)

        # Compute Live Mirror Coach feedback before applying action
        enc = encode_observation(obs)
        clone_probs = self.clone_model.action_distribution(obs)
        gto_raw = self.gto_bot.action_distribution(enc)
        legal_abs = legal_abstract_actions(obs)
        gto_tot = sum(gto_raw.get(a, 0.0) for a in legal_abs)
        gto_probs = (
            {a: round(gto_raw.get(a, 0.0) / gto_tot, 3) for a in legal_abs}
            if gto_tot > 0 else {a: round(1.0 / len(legal_abs), 3) for a in legal_abs}
        )
        rl_dist = self.exploiter_bot.action_distribution(enc)
        rl_best = max(legal_abs, key=lambda a: rl_dist.get(a, 0.0))
        gto_best = max(legal_abs, key=lambda a: gto_probs.get(a, 0.0))

        match_clone_pct = round(100.0 * clone_probs.get(abstract_act, 0.0), 1)
        if abstract_act == rl_best:
            note = f"Aligned with RL Exploiter ({rl_best}); GTO prefers {gto_best}."
        else:
            note = f"RL Exploiter favors {rl_best} here; GTO favors {gto_best}."

        self.coach_feedback.append({
            "street": obs.street,
            "your_action": abstract_act,
            "clone_probs": {a: round(clone_probs.get(a, 0.0), 3) for a in legal_abs},
            "gto_probs": gto_probs,
            "rl_exploit_note": note,
            "match_clone_pct": match_clone_pct,
        })

        # Log decision for JSON export
        added_est = 0 if sim_action.kind in ("fold", "check") else (
            obs.to_call if sim_action.kind == "call" else max(0, sim_action.amount - obs.bets[0])
        )
        is_vpip = obs.street == "preflop" and sim_action.kind in ("call", "bet", "raise") and not self._hero_vpip_done
        if is_vpip:
            self._hero_vpip_done = True
        is_pfr = obs.street == "preflop" and sim_action.kind in ("bet", "raise") and not self._hero_pfr_done
        if is_pfr:
            self._hero_pfr_done = True
        is_3b_opp = obs.street == "preflop" and self._preflop_raises == 1 and not self._hero_opened_pf
        is_3b = is_3b_opp and sim_action.kind == "raise"
        is_f3b_opp = obs.street == "preflop" and self._preflop_raises >= 2 and self._hero_opened_pf
        is_f3b = is_f3b_opp and sim_action.kind == "fold"
        is_cb_opp = obs.street == "flop" and self._preflop_aggressor == 0 and obs.to_call == 0 and not self._cbet_made
        is_cb = is_cb_opp and sim_action.kind == "bet"
        is_fcb_opp = obs.street == "flop" and self._cbet_made and obs.to_call > 0 and self._cbet_bettor != 0
        is_fcb = is_fcb_opp and sim_action.kind == "fold"

        self.hero_decisions_log.append({
            "street": obs.street,
            "action": abstract_act,
            "raw_action": sim_action.kind,
            "amount": added_est,
            "pot_before": obs.pot,
            "to_call": obs.to_call,
            "stack": obs.stacks[0],
            "position": position_name(0, self.button, self.num_players),
            "is_vpip": is_vpip,
            "is_pfr": is_pfr,
            "is_3bet_opp": is_3b_opp,
            "is_3bet": is_3b,
            "is_fold_to_3bet_opp": is_f3b_opp,
            "is_fold_to_3bet": is_f3b,
            "is_cbet_opp": is_cb_opp,
            "is_cbet": is_cb,
            "is_fold_to_cbet_opp": is_fcb_opp,
            "is_fold_to_cbet": is_fcb,
        })

        self.awaiting_hero = False
        self.current_obs = None
        self._apply_action(0, sim_action, obs)
        self._advance()
        return self.get_state()

    def export_hands_json(self) -> str:
        """Export completed hands in internal JSON format compatible with parse_hand_histories."""
        return json.dumps({"hands": self.completed_hands}, indent=2)

    def get_state(self) -> dict:
        """Return JSON-serializable snapshot of the current session state."""
        obs = self.current_obs
        return {
            "hand_id": self.hand_id,
            "mode": self.mode,
            "num_players": self.num_players,
            "street": self.street,
            "button": self.button,
            "pot": sum(self.contributions) if self.contributions else 0,
            "board": list(self.board),
            "hero_hole": list(self.holes[0]) if 0 in self.holes else [],
            "stacks": list(self.stacks),
            "bets": list(self.bets),
            "folded": [i in self.folded for i in range(self.num_players)],
            "seat_names": list(self.seat_names),
            "awaiting_hero": self.awaiting_hero,
            "hand_over": self.hand_over,
            "legal_actions": list(obs.legal_actions) if obs else [],
            "legal_abstract_actions": legal_abstract_actions(obs) if obs else [],
            "to_call": obs.to_call if obs else 0,
            "coach_feedback": list(self.coach_feedback),
            "completed_hands_count": len(self.completed_hands),
            "last_result": {
                "net": list(self.last_result.net),
                "payouts": list(self.last_result.payouts),
                "showdown": {str(k): list(v) for k, v in self.last_result.showdown.items()},
            } if self.last_result else None,
        }
