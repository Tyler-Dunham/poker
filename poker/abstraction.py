"""State and action abstractions for no-limit Hold'em RL and player modeling."""

from __future__ import annotations

from collections import Counter

from .simulator import RANKS, Action, Observation, hand_rank

ACTION_SPACE = ("fold", "check_call", "bet_small", "bet_medium", "bet_large")
STREETS = ("preflop", "flop", "turn", "river")
POSITIONS = ("EP", "MP", "CO", "BTN", "SB", "BB")
RANK_ORDER = "AKQJT98765432"

_RANK_IDX = {r: i for i, r in enumerate(RANK_ORDER)}
_RANK_VAL = {r: i + 2 for i, r in enumerate(RANKS)}


def canonical_preflop_hand(hole: tuple[str, str]) -> str:
    """Return canonical 169-hand code like 'AA', 'AKs', 'AKo', '72o'."""
    c1, c2 = hole
    r1, s1 = c1[0].upper(), c1[1].lower()
    r2, s2 = c2[0].upper(), c2[1].lower()
    if _RANK_IDX[r1] > _RANK_IDX[r2]:
        r1, r2 = r2, r1
    if r1 == r2:
        return f"{r1}{r2}"
    return f"{r1}{r2}{'s' if s1 == s2 else 'o'}"


def preflop_strength_bucket(hole: tuple[str, str]) -> int:
    """Map starting hand to strength bucket 0 (weakest) .. 5 (strongest)."""
    code = canonical_preflop_hand(hole)
    r1, r2 = code[0], code[1]
    suited = code.endswith("s")
    paired = r1 == r2
    v1, v2 = _RANK_VAL[r1], _RANK_VAL[r2]
    gap = v1 - v2

    # Chen-inspired score
    high_scores = {14: 10.0, 13: 8.0, 12: 7.0, 11: 6.0}
    score = high_scores.get(v1, v1 / 2.0)
    if paired:
        score = max(5.0, score * 2.0)
        if v1 == 5:
            score += 1.0
    else:
        if suited:
            score += 2.0
        if gap == 1:
            score += 1.0
        elif gap == 2:
            score -= 1.0
        elif gap == 3:
            score -= 2.0
        elif gap == 4:
            score -= 4.0
        elif gap >= 5:
            score -= 5.0
        if gap <= 2 and v1 < 12 and v2 < 12 and gap > 0:
            score += 1.0

    if score >= 13.0 or code in {"AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo", "AQs"}:
        return 5
    if score >= 10.0 or code in {"99", "88", "AQo", "AJs", "ATs", "KQs", "KJs", "QJs"}:
        return 4
    if score >= 7.5:
        return 3
    if score >= 5.5:
        return 2
    if score >= 3.0:
        return 1
    return 0


def _board_texture(board: tuple[str, ...]) -> str:
    """Classify board texture into 'dry', 'paired', or 'wet'."""
    ranks = [c[0].upper() for c in board]
    suits = [c[1].lower() for c in board]
    if len(set(ranks)) < len(ranks):
        return "paired"
    suit_counts = Counter(suits)
    max_suit = max(suit_counts.values(), default=0)
    vals = sorted(_RANK_VAL[r] for r in ranks)
    connected = sum(1 for i in range(len(vals) - 1) if vals[i + 1] - vals[i] <= 2)
    if max_suit >= 3 or (max_suit >= 2 and connected >= 2) or (len(vals) >= 3 and vals[-1] - vals[0] <= 4):
        return "wet"
    return "dry"


def _has_flush_draw(hole: tuple[str, str], board: tuple[str, ...]) -> bool:
    if len(board) >= 5:
        return False
    suits = [c[1].lower() for c in (*hole, *board)]
    hole_suits = {c[1].lower() for c in hole}
    counts = Counter(suits)
    return any(counts[s] == 4 for s in hole_suits)


def _has_straight_draw(hole: tuple[str, str], board: tuple[str, ...]) -> bool:
    if len(board) >= 5:
        return False
    vals = {_RANK_VAL[c[0].upper()] for c in (*hole, *board)}
    if 14 in vals:
        vals.add(1)
    for high in range(5, 15):
        window = set(range(high - 4, high + 1))
        if len(window & vals) == 4:
            return True
    return False


def postflop_strength_bucket(hole: tuple[str, str], board: tuple[str, ...]) -> tuple[int, str]:
    """Return (bucket 0..5, texture) for a postflop hand."""
    if len(board) < 3:
        return preflop_strength_bucket(hole), "dry"

    texture = _board_texture(board)
    rank_tuple = hand_rank((*hole, *board))
    cat = rank_tuple[0]

    board_vals = sorted((_RANK_VAL[c[0].upper()] for c in board), reverse=True)
    hole_vals = sorted((_RANK_VAL[c[0].upper()] for c in hole), reverse=True)
    board_counts = Counter(board_vals)

    if cat >= 5:
        bucket = 5
    elif cat == 4:
        bucket = 5
    elif cat == 3:
        trips_val = rank_tuple[1]
        if board_counts[trips_val] == 3:
            bucket = 2  # trips purely on board
        else:
            bucket = 5 if trips_val >= board_vals[0] else 4
    elif cat == 2:
        p1, p2 = rank_tuple[1], rank_tuple[2]
        if board_counts[p1] == 2 and board_counts[p2] == 2:
            bucket = 1
        elif board_counts[p1] == 2 or board_counts[p2] == 2:
            # One pair on board, one pair using hole card
            hero_pair = p2 if board_counts[p1] == 2 else p1
            bucket = 3 if hero_pair >= board_vals[0] else 2
        else:
            bucket = 4
    elif cat == 1:
        pair_val = rank_tuple[1]
        if board_counts[pair_val] == 2:
            # Board pair only -> high card strength
            bucket = 1 if hole_vals[0] > board_vals[0] else 0
        elif pair_val > board_vals[0]:
            bucket = 4  # overpair
        elif pair_val == board_vals[0]:
            kicker = max(v for v in hole_vals if v != pair_val) if hole_vals[0] != hole_vals[1] else hole_vals[0]
            bucket = 4 if kicker >= 12 else 3
        elif len(board_vals) >= 2 and pair_val >= board_vals[1]:
            bucket = 2
        else:
            bucket = 1
    else:
        overcards = sum(1 for v in hole_vals if v > board_vals[0])
        bucket = 1 if overcards == 2 else 0

    # Draw equity boost on flop and turn
    if len(board) < 5 and bucket < 4:
        f_draw = _has_flush_draw(hole, board)
        s_draw = _has_straight_draw(hole, board)
        if f_draw and s_draw:
            bucket = max(bucket, 4)
        elif f_draw or s_draw:
            bucket = min(4, bucket + 1)

    return max(0, min(5, bucket)), texture


def position_name(seat: int, button: int, n_players: int) -> str:
    """Map seat index relative to button to canonical position name."""
    if n_players <= 2:
        return "BTN" if seat == button else "BB"
    offset = (seat - button) % n_players
    if offset == 0:
        return "BTN"
    if offset == 1:
        return "SB"
    if offset == 2:
        return "BB"
    if offset == n_players - 1:
        return "CO"
    if offset == n_players - 2 and n_players >= 5:
        return "MP"
    return "EP"


def facing_bucket(to_call: int, pot: int) -> str:
    """Bucketize facing bet size relative to pot."""
    if to_call <= 0:
        return "none"
    ratio = to_call / max(1, pot)
    if ratio <= 0.35:
        return "small"
    if ratio <= 0.85:
        return "medium"
    return "large"


def spr_bucket(stack: int, pot: int) -> str:
    """Bucketize stack-to-pot ratio."""
    spr = stack / max(1, pot)
    if spr < 3.0:
        return "short"
    if spr < 8.0:
        return "medium"
    return "deep"


def build_encoded_state(
    street: str,
    position: str,
    hole: tuple[str, str],
    board: tuple[str, ...],
    to_call: int,
    pot: int,
    stack: int,
) -> dict:
    """Build encoded state dict from raw decision parameters."""
    hand_code = canonical_preflop_hand(hole)
    if street == "preflop" or len(board) < 3:
        hand_bucket = preflop_strength_bucket(hole)
        texture = "dry"
    else:
        hand_bucket, texture = postflop_strength_bucket(hole, board)

    facing = facing_bucket(to_call, pot)
    spr = spr_bucket(stack, pot)

    state_key = f"{street}|{position}|b{hand_bucket}|{facing}|{spr}|{texture}"
    coarse_key = f"{street}|{position}|b{hand_bucket}|{facing}"
    fallback_key = f"{street}|b{hand_bucket}|{facing}"

    facing_ratio = min(2.0, to_call / max(1, pot)) / 2.0
    spr_val = min(15.0, stack / max(1, pot)) / 15.0
    in_pos = 1.0 if position in ("BTN", "CO") else 0.0
    features = [
        1.0 if street == "preflop" else 0.0,
        1.0 if street == "flop" else 0.0,
        1.0 if street == "turn" else 0.0,
        1.0 if street == "river" else 0.0,
        hand_bucket / 5.0,
        facing_ratio,
        spr_val,
        in_pos,
        1.0 if to_call > 0 else 0.0,
        1.0 if texture == "dry" else 0.0,
        1.0 if texture == "paired" else 0.0,
        1.0 if texture == "wet" else 0.0,
        1.0,  # bias
    ]

    return {
        "state_key": state_key,
        "coarse_key": coarse_key,
        "fallback_key": fallback_key,
        "street": street,
        "position": position,
        "hand_bucket": hand_bucket,
        "hand_code": hand_code,
        "facing": facing,
        "spr": spr,
        "texture": texture,
        "features": features,
    }


def encode_observation(obs: Observation) -> dict:
    """Encode a simulator Observation into hierarchical keys and continuous features."""
    n_players = len(obs.stacks)
    pos = position_name(obs.seat, obs.button, n_players)
    active_villain_stacks = [
        obs.stacks[i] for i in range(n_players)
        if i != obs.seat and not obs.folded[i]
    ]
    eff_stack = min(obs.stacks[obs.seat], max(active_villain_stacks)) if active_villain_stacks else obs.stacks[obs.seat]
    return build_encoded_state(
        street=obs.street,
        position=pos,
        hole=obs.hole,
        board=obs.board,
        to_call=obs.to_call,
        pot=obs.pot,
        stack=eff_stack,
    )


def legal_abstract_actions(obs: Observation) -> list[str]:
    """Return list of abstract actions legal for this Observation."""
    actions: list[str] = []
    if "fold" in obs.legal_actions:
        actions.append("fold")
    if "check" in obs.legal_actions or "call" in obs.legal_actions:
        actions.append("check_call")
    if (("bet" in obs.legal_actions or "raise" in obs.legal_actions)
            and obs.min_raise_to is not None and obs.max_raise_to is not None):
        actions.extend(("bet_small", "bet_medium", "bet_large"))
    return actions


def abstract_to_legal_action(abstract_action: str, obs: Observation) -> Action:
    """Convert an abstract action name into a strictly legal simulator Action."""
    if abstract_action == "fold":
        if "fold" in obs.legal_actions:
            return Action("fold")
        return Action("call" if obs.to_call > 0 else "check")

    if abstract_action == "check_call":
        return Action("call" if obs.to_call > 0 else "check")

    can_bet_or_raise = (
        ("bet" in obs.legal_actions or "raise" in obs.legal_actions)
        and obs.min_raise_to is not None
        and obs.max_raise_to is not None
    )
    if not can_bet_or_raise:
        return Action("call" if obs.to_call > 0 else "check")

    raise_kind = "bet" if "bet" in obs.legal_actions else "raise"
    min_to = obs.min_raise_to
    max_to = obs.max_raise_to
    assert min_to is not None and max_to is not None

    if min_to == max_to:
        return Action(raise_kind, min_to)

    current_bet = obs.bets[obs.seat] + obs.to_call
    pot_after_call = max(1, obs.pot + obs.to_call)

    if abstract_action == "bet_small":
        target = current_bet + 0.35 * pot_after_call
    elif abstract_action == "bet_medium":
        target = current_bet + 0.66 * pot_after_call
    else:  # bet_large
        target = current_bet + 1.20 * pot_after_call

    amount = max(min_to, min(max_to, int(round(target))))
    return Action(raise_kind, amount)
