"""Deterministic realistic sample hand history generator for player archetypes."""

from __future__ import annotations

from random import Random

from .abstraction import preflop_strength_bucket, postflop_strength_bucket
from .simulator import RANKS, SUITS, hand_rank

SAMPLE_PACKS: dict[str, dict] = {
    "calling_station": {
        "id": "calling_station",
        "name": "Calling Station (Loose-Passive)",
        "description": "High VPIP (~42%), low PFR (~11%), rarely folds to c-bets (~24%), low aggression factor (~0.7).",
        "target_stats": {"vpip": 42.0, "pfr": 11.0, "fold_to_cbet": 24.0, "af": 0.7},
    },
    "over_folding_nit": {
        "id": "over_folding_nit",
        "name": "Over-Folding Nit (Tight-Weak)",
        "description": "Tight VPIP (~15%), PFR (~12%), folds too often to 3-bets (~78%) and flop c-bets (~72%).",
        "target_stats": {"vpip": 15.0, "pfr": 12.0, "fold_to_3bet": 78.0, "fold_to_cbet": 72.0},
    },
    "aggro_grinder": {
        "id": "aggro_grinder",
        "name": "Aggro Grinder (LAG)",
        "description": "Loose-aggressive VPIP (~30%), PFR (~25%), high 3-bet (~14%), high flop c-bet (~82%), over-bluffs rivers.",
        "target_stats": {"vpip": 30.0, "pfr": 25.0, "three_bet": 14.0, "cbet_flop": 82.0},
    },
}

_VILLAINS = ["Villain_A", "Villain_B", "Villain_C", "Villain_D", "Villain_E"]


def _fmt_chips(cents: int) -> str:
    return f"${cents / 100:.2f}"


def generate_sample_hand_history(pack_id: str, hero_name: str = "Hero", num_hands: int = 180) -> str:
    """Generate deterministic Global Poker / PokerStars text hand histories for a target archetype."""
    if pack_id not in SAMPLE_PACKS:
        raise ValueError(f"Unknown sample pack: {pack_id!r}")

    rng = Random(f"{pack_id}:{hero_name}:{num_hands}")
    sb_amt = 5
    bb_amt = 10
    start_stack = 1000
    hands_out: list[str] = []

    for h_idx in range(1, num_hands + 1):
        deck = [r + s for r in RANKS for s in SUITS]
        rng.shuffle(deck)

        # Rotate button across 6 seats; Seat 1 is hero_name
        btn_seat = ((h_idx - 1) % 6) + 1
        players = [hero_name] + _VILLAINS
        holes = {p: (deck.pop(), deck.pop()) for p in players}

        sb_seat = (btn_seat % 6) + 1
        bb_seat = (sb_seat % 6) + 1
        sb_player = players[sb_seat - 1]
        bb_player = players[bb_seat - 1]

        stacks = {p: start_stack for p in players}
        contrib = {p: 0 for p in players}
        street_bets = {p: 0 for p in players}
        folded: set[str] = set()

        lines: list[str] = [
            f"Global Poker Hand #{100000 + h_idx}: Hold'em No Limit ($0.05/$0.10 USD) - 2026/11/20 12:{h_idx % 60:02d}:00 ET",
            f"Table 'Pacific {pack_id}' 6-max Seat #{btn_seat} is the button",
        ]
        for s_num, p_name in enumerate(players, start=1):
            lines.append(f"Seat {s_num}: {p_name} ({_fmt_chips(start_stack)} in chips)")

        def post(p: str, amt: int, label: str) -> None:
            stacks[p] -= amt
            contrib[p] += amt
            street_bets[p] += amt
            lines.append(f"{p}: posts {label} {_fmt_chips(amt)}")

        post(sb_player, sb_amt, "small blind")
        post(bb_player, bb_amt, "big blind")
        lines.append("*** HOLE CARDS ***")
        lines.append(f"Dealt to {hero_name} [{holes[hero_name][0]} {holes[hero_name][1]}]")

        # Simplified realistic 6-max preflop -> postflop flow where Hero and 1 primary villain contest most pots
        # Choose active villain (different from hero)
        utg_idx = bb_seat % 6  # 0-based index
        order_pf = [players[(utg_idx + k) % 6] for k in range(6)]
        hero_pos_idx = order_pf.index(hero_name)

        pf_bucket = preflop_strength_bucket(holes[hero_name])
        villain_focus = players[(players.index(hero_name) + 2) % 6]

        # Decide whether an earlier villain opens before hero
        villain_opened = False
        opener: str | None = None
        current_bet = bb_amt
        pf_aggressor: str | None = None

        # Pre-hero actors
        for p in order_pf[:hero_pos_idx]:
            if p == villain_focus and rng.random() < 0.30:
                raise_to = 30
                added = raise_to - street_bets[p]
                stacks[p] -= added
                contrib[p] += added
                street_bets[p] = raise_to
                current_bet = raise_to
                villain_opened = True
                opener = p
                pf_aggressor = p
                lines.append(f"{p}: raises {_fmt_chips(20)} to {_fmt_chips(raise_to)}")
            else:
                folded.add(p)
                lines.append(f"{p}: folds")

        # Hero preflop decision
        to_call_pf = current_bet - street_bets[hero_name]
        hero_acted_pf = "fold"

        if pack_id == "calling_station":
            if pf_bucket >= 5 or (pf_bucket == 4 and rng.random() < 0.55):
                hero_acted_pf = "raise"
            elif pf_bucket >= 2 or (pf_bucket == 1 and rng.random() < 0.65):
                hero_acted_pf = "call" if to_call_pf > 0 else "check"
            else:
                hero_acted_pf = "fold" if to_call_pf > 0 else "check"
        elif pack_id == "over_folding_nit":
            if pf_bucket >= 4 or (pf_bucket == 3 and not villain_opened and rng.random() < 0.55):
                hero_acted_pf = "raise"
            elif villain_opened and ((pf_bucket == 3 and rng.random() < 0.55) or (pf_bucket == 2 and rng.random() < 0.25)):
                hero_acted_pf = "call"
            else:
                hero_acted_pf = "fold" if to_call_pf > 0 else "check"
        else:  # aggro_grinder
            if villain_opened:
                if pf_bucket >= 4 or (pf_bucket == 3 and rng.random() < 0.45) or (pf_bucket == 2 and rng.random() < 0.25):
                    hero_acted_pf = "raise"
                elif pf_bucket >= 2 and rng.random() < 0.40:
                    hero_acted_pf = "call"
                else:
                    hero_acted_pf = "fold"
            else:
                if pf_bucket >= 3 or (pf_bucket == 2 and rng.random() < 0.55):
                    hero_acted_pf = "raise"
                elif pf_bucket == 1 and rng.random() < 0.12:
                    hero_acted_pf = "call" if to_call_pf > 0 else "check"
                else:
                    hero_acted_pf = "fold" if to_call_pf > 0 else "check"

        if hero_acted_pf == "fold":
            folded.add(hero_name)
            lines.append(f"{hero_name}: folds")
        elif hero_acted_pf in ("check", "call"):
            if to_call_pf == 0:
                lines.append(f"{hero_name}: checks")
            else:
                stacks[hero_name] -= to_call_pf
                contrib[hero_name] += to_call_pf
                street_bets[hero_name] = current_bet
                lines.append(f"{hero_name}: calls {_fmt_chips(to_call_pf)}")
        else:
            raise_to = 90 if villain_opened else 30
            added = raise_to - street_bets[hero_name]
            stacks[hero_name] -= added
            contrib[hero_name] += added
            street_bets[hero_name] = raise_to
            current_bet = raise_to
            pf_aggressor = hero_name
            lines.append(f"{hero_name}: raises {_fmt_chips(raise_to - bb_amt)} to {_fmt_chips(raise_to)}")

        # Remaining players after hero
        active_villain: str | None = opener if (opener and opener not in folded) else None
        for p in order_pf[hero_pos_idx + 1:]:
            if hero_name in folded:
                if active_villain is None and p != hero_name:
                    active_villain = p
                    if current_bet > street_bets[p]:
                        added = current_bet - street_bets[p]
                        stacks[p] -= added
                        contrib[p] += added
                        street_bets[p] = current_bet
                        lines.append(f"{p}: calls {_fmt_chips(added)}")
                    else:
                        lines.append(f"{p}: checks")
                else:
                    folded.add(p)
                    lines.append(f"{p}: folds")
            elif active_villain is None and p != hero_name:
                # Villain decides whether to 3-bet (if hero opened) or call
                if hero_acted_pf == "raise":
                    three_bet_prob = 0.22 if pack_id == "over_folding_nit" else 0.12
                    if rng.random() < three_bet_prob:
                        raise_to = 90
                        added = raise_to - street_bets[p]
                        stacks[p] -= added
                        contrib[p] += added
                        street_bets[p] = raise_to
                        current_bet = raise_to
                        pf_aggressor = p
                        active_villain = p
                        lines.append(f"{p}: raises {_fmt_chips(60)} to {_fmt_chips(raise_to)}")
                    elif rng.random() < 0.65:
                        added = current_bet - street_bets[p]
                        stacks[p] -= added
                        contrib[p] += added
                        street_bets[p] = current_bet
                        active_villain = p
                        lines.append(f"{p}: calls {_fmt_chips(added)}")
                    else:
                        folded.add(p)
                        lines.append(f"{p}: folds")
                else:
                    added = current_bet - street_bets[p]
                    if added == 0:
                        active_villain = p
                        lines.append(f"{p}: checks")
                    elif rng.random() < 0.65:
                        stacks[p] -= added
                        contrib[p] += added
                        street_bets[p] = current_bet
                        active_villain = p
                        lines.append(f"{p}: calls {_fmt_chips(added)}")
                    else:
                        folded.add(p)
                        lines.append(f"{p}: folds")
            else:
                folded.add(p)
                lines.append(f"{p}: folds")

        # If villain 3-bet hero, hero responds
        if hero_name not in folded and pf_aggressor is not None and pf_aggressor != hero_name and current_bet == 90 and street_bets[hero_name] < 90:
            to_call_3b = 90 - street_bets[hero_name]
            fold_3b_prob = 0.80 if pack_id == "over_folding_nit" else (0.30 if pack_id == "calling_station" else 0.40)
            if pf_bucket >= 5 and pack_id != "over_folding_nit":
                fold_3b_prob = 0.05
            elif pf_bucket >= 5 and pack_id == "over_folding_nit":
                fold_3b_prob = 0.25
            if rng.random() < fold_3b_prob:
                folded.add(hero_name)
                lines.append(f"{hero_name}: folds")
            else:
                stacks[hero_name] -= to_call_3b
                contrib[hero_name] += to_call_3b
                street_bets[hero_name] = 90
                lines.append(f"{hero_name}: calls {_fmt_chips(to_call_3b)}")

        # Ensure opener also calls if hero 3-bet
        if opener and opener not in folded and street_bets[opener] < current_bet:
            if rng.random() < 0.65:
                added = current_bet - street_bets[opener]
                stacks[opener] -= added
                contrib[opener] += added
                street_bets[opener] = current_bet
                lines.append(f"{opener}: calls {_fmt_chips(added)}")
            else:
                folded.add(opener)
                lines.append(f"{opener}: folds")

        contenders = [p for p in players if p not in folded]
        board: list[str] = []

        # Postflop streets if >= 2 contenders and hero is in
        if len(contenders) >= 2 and hero_name in contenders:
            villain = next(p for p in contenders if p != hero_name)
            # Collapse any extra contenders to keep clean HU postflop
            for extra in list(contenders):
                if extra not in (hero_name, villain):
                    folded.add(extra)
            contenders = [hero_name, villain]

            flop = [deck.pop(), deck.pop(), deck.pop()]
            turn = deck.pop()
            river = deck.pop()

            for street_name, new_cards in (("FLOP", flop), ("TURN", [turn]), ("RIVER", [river])):
                if len(contenders) < 2:
                    break
                board.extend(new_cards)
                if street_name == "FLOP":
                    lines.append(f"*** FLOP *** [{' '.join(board)}]")
                elif street_name == "TURN":
                    lines.append(f"*** TURN *** [{' '.join(board[:3])}] [{board[3]}]")
                else:
                    lines.append(f"*** RIVER *** [{' '.join(board[:4])}] [{board[4]}]")

                pot_now = sum(contrib.values())
                hero_sb, _ = postflop_strength_bucket(holes[hero_name], tuple(board))
                # Determine acting order postflop (SB/BB acts first)
                pf_order = [players[(btn_seat + k) % 6] for k in range(6)]
                act_first = hero_name if pf_order.index(hero_name) < pf_order.index(villain) else villain
                act_second = villain if act_first == hero_name else hero_name

                bet_size = max(bb_amt, int(round(pot_now * 0.6)))
                if street_name == "FLOP" and pf_aggressor == villain:
                    # Villain c-bets high frequency
                    if act_first == hero_name:
                        lines.append(f"{hero_name}: checks")
                        v_bet = min(bet_size, stacks[villain])
                        stacks[villain] -= v_bet
                        contrib[villain] += v_bet
                        lines.append(f"{villain}: bets {_fmt_chips(v_bet)}")
                        # Hero faces c-bet
                        if pack_id == "calling_station":
                            fold_prob = 0.24 if hero_sb < 2 else 0.05
                        elif pack_id == "over_folding_nit":
                            fold_prob = 0.75 if hero_sb < 3 else 0.15
                        else:
                            fold_prob = 0.45 if hero_sb < 2 else 0.10
                        if rng.random() < fold_prob:
                            folded.add(hero_name)
                            lines.append(f"{hero_name}: folds")
                            contenders = [villain]
                        else:
                            c_amt = min(v_bet, stacks[hero_name])
                            stacks[hero_name] -= c_amt
                            contrib[hero_name] += c_amt
                            lines.append(f"{hero_name}: calls {_fmt_chips(c_amt)}")
                    else:
                        v_bet = min(bet_size, stacks[villain])
                        stacks[villain] -= v_bet
                        contrib[villain] += v_bet
                        lines.append(f"{villain}: bets {_fmt_chips(v_bet)}")
                        if pack_id == "calling_station":
                            fold_prob = 0.24 if hero_sb < 2 else 0.05
                        elif pack_id == "over_folding_nit":
                            fold_prob = 0.75 if hero_sb < 3 else 0.15
                        else:
                            fold_prob = 0.45 if hero_sb < 2 else 0.10
                        if rng.random() < fold_prob:
                            folded.add(hero_name)
                            lines.append(f"{hero_name}: folds")
                            contenders = [villain]
                        else:
                            c_amt = min(v_bet, stacks[hero_name])
                            stacks[hero_name] -= c_amt
                            contrib[hero_name] += c_amt
                            lines.append(f"{hero_name}: calls {_fmt_chips(c_amt)}")
                else:
                    # Hero initiative or checked street
                    hero_bets = False
                    if pack_id == "aggro_grinder":
                        if street_name == "FLOP" and pf_aggressor == hero_name:
                            hero_bets = rng.random() < 0.82
                        elif street_name == "RIVER":
                            hero_bets = hero_sb >= 3 or rng.random() < 0.48
                        else:
                            hero_bets = hero_sb >= 2 or rng.random() < 0.40
                    elif pack_id == "calling_station":
                        hero_bets = hero_sb >= 5 and rng.random() < 0.25
                    else:
                        hero_bets = hero_sb >= 3 and rng.random() < 0.55

                    if act_first == villain:
                        # Villain occasionally bets into calling_station so calling_station can call
                        if pack_id == "calling_station" and rng.random() < 0.50 and stacks[villain] >= bb_amt:
                            v_bet = min(bet_size, stacks[villain])
                            stacks[villain] -= v_bet
                            contrib[villain] += v_bet
                            lines.append(f"{villain}: bets {_fmt_chips(v_bet)}")
                            if rng.random() < 0.22:
                                folded.add(hero_name)
                                lines.append(f"{hero_name}: folds")
                                contenders = [villain]
                            else:
                                c_amt = min(v_bet, stacks[hero_name])
                                stacks[hero_name] -= c_amt
                                contrib[hero_name] += c_amt
                                lines.append(f"{hero_name}: calls {_fmt_chips(c_amt)}")
                            continue
                        lines.append(f"{villain}: checks")
                    if hero_bets and stacks[hero_name] >= bb_amt:
                        h_bet = min(bet_size, stacks[hero_name])
                        stacks[hero_name] -= h_bet
                        contrib[hero_name] += h_bet
                        lines.append(f"{hero_name}: bets {_fmt_chips(h_bet)}")
                        if rng.random() < 0.42:
                            folded.add(villain)
                            lines.append(f"{villain}: folds")
                            contenders = [hero_name]
                        else:
                            vc = min(h_bet, stacks[villain])
                            stacks[villain] -= vc
                            contrib[villain] += vc
                            lines.append(f"{villain}: calls {_fmt_chips(vc)}")
                    else:
                        lines.append(f"{hero_name}: checks")
                        if act_second == villain:
                            lines.append(f"{villain}: checks")

        # Award pot
        total_pot = sum(contrib.values())
        if len(contenders) == 1:
            winner = contenders[0]
            lines.append(f"{winner} collected {_fmt_chips(total_pot)} from pot")
        else:
            lines.append("*** SHOW DOWN ***")
            r_map = {p: hand_rank((*holes[p], *board)) for p in contenders}
            best = max(r_map.values())
            winners = [p for p in contenders if r_map[p] == best]
            for p in contenders:
                lines.append(f"{p}: shows [{holes[p][0]} {holes[p][1]}]")
            share = total_pot // len(winners)
            for w in winners:
                lines.append(f"{w} collected {_fmt_chips(share)} from pot")

        lines.append("*** SUMMARY ***")
        lines.append(f"Total pot {_fmt_chips(total_pot)} | Rake $0.00")
        if board:
            lines.append(f"Board [{' '.join(board)}]")

        hands_out.append("\n".join(lines))

    return "\n\n".join(hands_out)
