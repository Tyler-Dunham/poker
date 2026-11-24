"""Multi-format poker hand history parser and player statistics calculator."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from .abstraction import (
    ACTION_SPACE,
    POSITIONS,
    RANK_ORDER,
    STREETS,
    build_encoded_state,
    canonical_preflop_hand,
    position_name,
)

_SUIT_UNICODE = {
    "♠": "s",
    "♥": "h",
    "♦": "d",
    "♣": "c",
}

_ALL_169_HANDS: list[str] = []
for _i, _r1 in enumerate(RANK_ORDER):
    for _j, _r2 in enumerate(RANK_ORDER):
        if _i == _j:
            _ALL_169_HANDS.append(f"{_r1}{_r2}")
        elif _i < _j:
            _ALL_169_HANDS.append(f"{_r1}{_r2}s")
        else:
            _ALL_169_HANDS.append(f"{_r2}{_r1}o")


@dataclass
class ParsedDecision:
    street: str
    action: str
    raw_action: str
    amount: int
    pot_before: int
    to_call: int
    position: str
    hand_bucket: int
    hand_code: str
    facing: str
    spr: str
    texture: str
    state_key: str
    coarse_key: str
    fallback_key: str
    features: list[float] = field(default_factory=list)
    is_vpip: bool = False
    is_pfr: bool = False
    is_3bet_opp: bool = False
    is_3bet: bool = False
    is_fold_to_3bet_opp: bool = False
    is_fold_to_3bet: bool = False
    is_cbet_opp: bool = False
    is_cbet: bool = False
    is_fold_to_cbet_opp: bool = False
    is_fold_to_cbet: bool = False


@dataclass
class ParsedHand:
    hand_id: str
    source_format: str
    hero_name: str
    hero_hole: tuple[str, str]
    position: str
    board: tuple[str, ...]
    big_blind: int
    net_chips: int
    net_bb: float
    went_to_showdown: bool
    won_at_showdown: bool
    decisions: list[ParsedDecision] = field(default_factory=list)


def normalize_card(token: str) -> str:
    """Normalize a card string like '10s' or 'A♠' to 'Ts' or 'As'."""
    t = token.strip()
    for uni, ascii_s in _SUIT_UNICODE.items():
        t = t.replace(uni, ascii_s)
    if t.startswith("10"):
        t = "T" + t[2:]
    if len(t) != 2:
        raise ValueError(f"Invalid card token: {token!r}")
    return t[0].upper() + t[1].lower()


def _parse_cards_bracket(text: str) -> tuple[str, ...]:
    tokens = text.strip().split()
    return tuple(normalize_card(tok) for tok in tokens if tok)


def _to_chips(val_str: str, scale: int) -> int:
    cleaned = val_str.replace("$", "").replace(",", "").strip()
    return int(round(float(cleaned) * scale))


def _raw_to_abstract(raw_action: str, added_amount: int, to_call: int, pot_before: int) -> str:
    if raw_action == "fold":
        return "fold"
    if raw_action in ("check", "call"):
        return "check_call"
    pot_after_call = max(1, pot_before + to_call)
    raise_portion = max(0, added_amount - to_call)
    ratio = raise_portion / pot_after_call
    if ratio <= 0.48:
        return "bet_small"
    if ratio <= 0.90:
        return "bet_medium"
    return "bet_large"


def _detect_format(block: str) -> str:
    head = block.lstrip()[:120]
    if "Global Poker" in head:
        return "global_poker"
    if "GGPoker" in head or head.startswith("Poker Hand #HD"):
        return "ggpoker"
    if "PokerStars" in head:
        return "pokerstars"
    return "global_poker"


def _parse_json_hands(data: object, hero_name: str | None) -> list[ParsedHand]:
    items = data.get("hands", []) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("JSON hand history must be a list or an object with 'hands'")

    parsed: list[ParsedHand] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        h_name = hero_name or str(item.get("hero_name") or "Hero")
        hole_raw = item.get("hero_hole") or item.get("hole") or ["As", "Kd"]
        hero_hole = (normalize_card(hole_raw[0]), normalize_card(hole_raw[1]))
        board_raw = item.get("board") or []
        board = tuple(normalize_card(c) for c in board_raw)
        bb = int(item.get("big_blind", 10) or 10)
        net_chips = int(item.get("net_chips", 0))
        net_bb = float(item.get("net_bb", net_chips / max(1, bb)))
        pos = str(item.get("position", "BTN"))

        decisions: list[ParsedDecision] = []
        for dec in item.get("decisions", []):
            street = str(dec.get("street", "preflop"))
            raw_action = str(dec.get("raw_action", dec.get("action", "check")))
            amount = int(dec.get("amount", 0))
            pot_before = int(dec.get("pot_before", bb))
            to_call = int(dec.get("to_call", 0))
            stack = int(dec.get("stack", bb * 100))
            dec_pos = str(dec.get("position", pos))
            sub_board = board[:3] if street == "flop" else (board[:4] if street == "turn" else (board[:5] if street == "river" else ()))
            enc = build_encoded_state(street, dec_pos, hero_hole, sub_board, to_call, pot_before, stack)
            abstract = str(dec.get("action", ""))
            if abstract not in ACTION_SPACE:
                abstract = _raw_to_abstract(raw_action, amount, to_call, pot_before)

            decisions.append(ParsedDecision(
                street=street,
                action=abstract,
                raw_action=raw_action,
                amount=amount,
                pot_before=pot_before,
                to_call=to_call,
                position=dec_pos,
                hand_bucket=enc["hand_bucket"],
                hand_code=enc["hand_code"],
                facing=enc["facing"],
                spr=enc["spr"],
                texture=enc["texture"],
                state_key=enc["state_key"],
                coarse_key=enc["coarse_key"],
                fallback_key=enc["fallback_key"],
                features=enc["features"],
                is_vpip=bool(dec.get("is_vpip", False)),
                is_pfr=bool(dec.get("is_pfr", False)),
                is_3bet_opp=bool(dec.get("is_3bet_opp", False)),
                is_3bet=bool(dec.get("is_3bet", False)),
                is_fold_to_3bet_opp=bool(dec.get("is_fold_to_3bet_opp", False)),
                is_fold_to_3bet=bool(dec.get("is_fold_to_3bet", False)),
                is_cbet_opp=bool(dec.get("is_cbet_opp", False)),
                is_cbet=bool(dec.get("is_cbet", False)),
                is_fold_to_cbet_opp=bool(dec.get("is_fold_to_cbet_opp", False)),
                is_fold_to_cbet=bool(dec.get("is_fold_to_cbet", False)),
            ))

        parsed.append(ParsedHand(
            hand_id=str(item.get("hand_id", f"json_{idx + 1}")),
            source_format="json",
            hero_name=h_name,
            hero_hole=hero_hole,
            position=pos,
            board=board,
            big_blind=bb,
            net_chips=net_chips,
            net_bb=net_bb,
            went_to_showdown=bool(item.get("went_to_showdown", False)),
            won_at_showdown=bool(item.get("won_at_showdown", False)),
            decisions=decisions,
        ))
    return parsed


_HAND_ID_RE = re.compile(r"Hand #([A-Za-z0-9\-]+)")
_BLINDS_RE = re.compile(r"\(\$?([0-9]+(?:\.[0-9]+)?)/\$?([0-9]+(?:\.[0-9]+)?)\)")
_BUTTON_RE = re.compile(r"Seat #(\d+) is the button")
_SEAT_RE = re.compile(r"^Seat (\d+):\s+(.+?)\s+\(\$?([0-9,]+(?:\.[0-9]+)?)\s+in chips\)")
_DEALT_RE = re.compile(r"^Dealt to (.+?) \[([^\]]+)\]")
_BLIND_POST_RE = re.compile(r"^(.+?): posts (?:small|big) blind \$?([0-9,]+(?:\.[0-9]+)?)")
_ACTION_RE = re.compile(
    r"^(.+?): (folds|checks|calls|bets|raises)(?:\s+\$?([0-9,]+(?:\.[0-9]+)?))?(?:\s+to\s+\$?([0-9,]+(?:\.[0-9]+)?))?"
)
_COLLECTED_RE = re.compile(r"^(.+?)\s+collected\s+\$?([0-9,]+(?:\.[0-9]+)?)")
_UNCALLED_RE = re.compile(r"^Uncalled bet \(\$?([0-9,]+(?:\.[0-9]+)?)\) returned to (.+?)$")
_BRACKET_CARDS_RE = re.compile(r"\[([^\]]+)\]")


def _parse_text_hand(block: str, hero_name: str | None) -> ParsedHand | None:
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if not lines:
        return None

    source_format = _detect_format(block)
    header = lines[0]
    m_id = _HAND_ID_RE.search(header)
    hand_id = m_id.group(1) if m_id else "unknown"

    m_blinds = _BLINDS_RE.search(header)
    if m_blinds:
        raw_sb, raw_bb = m_blinds.group(1), m_blinds.group(2)
        scale = 100 if ("." in raw_sb or "." in raw_bb or "$" in header) else 1
        bb = max(1, _to_chips(raw_bb, scale))
    else:
        scale = 100 if "$" in block else 1
        bb = 10

    button_seat_num = 1
    seats: dict[int, str] = {}
    stacks: dict[str, int] = {}
    detected_hero: str | None = hero_name

    for ln in lines:
        m_btn = _BUTTON_RE.search(ln)
        if m_btn:
            button_seat_num = int(m_btn.group(1))
        m_seat = _SEAT_RE.match(ln)
        if m_seat:
            s_num = int(m_seat.group(1))
            p_name = m_seat.group(2).strip()
            s_chips = _to_chips(m_seat.group(3), scale)
            seats[s_num] = p_name
            stacks[p_name] = s_chips
        m_dealt = _DEALT_RE.match(ln)
        if m_dealt:
            if detected_hero is None:
                detected_hero = m_dealt.group(1).strip()

    if not detected_hero or detected_hero not in stacks:
        return None

    sorted_seat_nums = sorted(seats.keys())
    seat_index_by_name = {seats[s_num]: idx for idx, s_num in enumerate(sorted_seat_nums)}
    btn_idx = seat_index_by_name.get(seats.get(button_seat_num, ""), 0)
    hero_idx = seat_index_by_name[detected_hero]
    hero_pos = position_name(hero_idx, btn_idx, len(sorted_seat_nums))

    hero_hole: tuple[str, str] | None = None
    board: list[str] = []
    street = "preflop"
    pot = 0
    street_bets: dict[str, int] = {name: 0 for name in stacks}
    total_contrib: dict[str, int] = {name: 0 for name in stacks}
    collected: dict[str, int] = {name: 0 for name in stacks}

    preflop_raises = 0
    preflop_aggressor: str | None = None
    hero_opened_pf = False
    hero_vpip_done = False
    hero_pfr_done = False
    cbet_made_on_flop = False
    cbet_bettor: str | None = None
    went_to_showdown = False
    won_at_showdown = False

    decisions: list[ParsedDecision] = []

    for ln in lines:
        if ln.startswith("*** FLOP ***"):
            street = "flop"
            street_bets = {name: 0 for name in stacks}
            brackets = _BRACKET_CARDS_RE.findall(ln)
            if brackets:
                board = list(_parse_cards_bracket(brackets[0]))
            continue
        if ln.startswith("*** TURN ***"):
            street = "turn"
            street_bets = {name: 0 for name in stacks}
            brackets = _BRACKET_CARDS_RE.findall(ln)
            if brackets:
                board.extend(_parse_cards_bracket(brackets[-1]))
            continue
        if ln.startswith("*** RIVER ***"):
            street = "river"
            street_bets = {name: 0 for name in stacks}
            brackets = _BRACKET_CARDS_RE.findall(ln)
            if brackets:
                board.extend(_parse_cards_bracket(brackets[-1]))
            continue
        if ln.startswith("*** SHOW DOWN ***"):
            went_to_showdown = True
            continue
        if ln.startswith("*** SUMMARY ***"):
            break

        m_dealt = _DEALT_RE.match(ln)
        if m_dealt and m_dealt.group(1).strip() == detected_hero:
            cards = _parse_cards_bracket(m_dealt.group(2))
            if len(cards) == 2:
                hero_hole = (cards[0], cards[1])
            continue

        m_blind = _BLIND_POST_RE.match(ln)
        if m_blind:
            p_name = m_blind.group(1).strip()
            amt = _to_chips(m_blind.group(2), scale)
            if p_name in stacks:
                amt = min(amt, stacks[p_name])
                stacks[p_name] -= amt
                street_bets[p_name] += amt
                total_contrib[p_name] += amt
                pot += amt
            continue

        m_uncalled = _UNCALLED_RE.match(ln)
        if m_uncalled:
            amt = _to_chips(m_uncalled.group(1), scale)
            p_name = m_uncalled.group(2).strip()
            if p_name in collected:
                collected[p_name] += amt
                pot = max(0, pot - amt)
            continue

        m_coll = _COLLECTED_RE.match(ln)
        if m_coll:
            p_name = m_coll.group(1).strip()
            amt = _to_chips(m_coll.group(2), scale)
            if p_name in collected:
                collected[p_name] += amt
                if went_to_showdown and p_name == detected_hero:
                    won_at_showdown = True
            continue

        m_act = _ACTION_RE.match(ln)
        if not m_act:
            continue

        p_name = m_act.group(1).strip()
        if p_name not in stacks:
            continue
        verb = m_act.group(2)
        raw_action = {"folds": "fold", "checks": "check", "calls": "call", "bets": "bet", "raises": "raise"}[verb]

        current_max = max(street_bets.values(), default=0)
        to_call = max(0, current_max - street_bets[p_name])
        pot_before = pot

        if raw_action in ("fold", "check"):
            added = 0
        elif raw_action == "call":
            added = _to_chips(m_act.group(3), scale) if m_act.group(3) else to_call
            added = min(added, stacks[p_name])
        elif raw_action == "bet":
            bet_amt = _to_chips(m_act.group(3), scale) if m_act.group(3) else bb
            added = min(bet_amt, stacks[p_name])
        else:  # raise
            if m_act.group(4):
                raise_to = _to_chips(m_act.group(4), scale)
                added = max(0, raise_to - street_bets[p_name])
            elif m_act.group(3):
                added = _to_chips(m_act.group(3), scale)
            else:
                added = to_call + bb
            added = min(added, stacks[p_name])

        if p_name == detected_hero and hero_hole is not None:
            enc = build_encoded_state(
                street=street,
                position=hero_pos,
                hole=hero_hole,
                board=tuple(board),
                to_call=to_call,
                pot=max(1, pot_before),
                stack=stacks[p_name],
            )
            abstract = _raw_to_abstract(raw_action, added, to_call, max(1, pot_before))

            is_vpip = False
            is_pfr = False
            is_3bet_opp = False
            is_3bet = False
            is_fold_to_3bet_opp = False
            is_fold_to_3bet = False
            is_cbet_opp = False
            is_cbet = False
            is_fold_to_cbet_opp = False
            is_fold_to_cbet = False

            if street == "preflop":
                if raw_action in ("call", "bet", "raise") and not hero_vpip_done:
                    is_vpip = True
                    hero_vpip_done = True
                if raw_action in ("bet", "raise") and not hero_pfr_done:
                    is_pfr = True
                    hero_pfr_done = True
                if preflop_raises == 1 and not hero_opened_pf:
                    is_3bet_opp = True
                    if raw_action == "raise":
                        is_3bet = True
                if preflop_raises >= 2 and hero_opened_pf:
                    is_fold_to_3bet_opp = True
                    if raw_action == "fold":
                        is_fold_to_3bet = True
            elif street == "flop":
                if preflop_aggressor == detected_hero and to_call == 0 and not cbet_made_on_flop:
                    is_cbet_opp = True
                    if raw_action == "bet":
                        is_cbet = True
                elif cbet_made_on_flop and to_call > 0 and cbet_bettor != detected_hero:
                    is_fold_to_cbet_opp = True
                    if raw_action == "fold":
                        is_fold_to_cbet = True

            decisions.append(ParsedDecision(
                street=street,
                action=abstract,
                raw_action=raw_action,
                amount=added,
                pot_before=pot_before,
                to_call=to_call,
                position=hero_pos,
                hand_bucket=enc["hand_bucket"],
                hand_code=enc["hand_code"],
                facing=enc["facing"],
                spr=enc["spr"],
                texture=enc["texture"],
                state_key=enc["state_key"],
                coarse_key=enc["coarse_key"],
                fallback_key=enc["fallback_key"],
                features=enc["features"],
                is_vpip=is_vpip,
                is_pfr=is_pfr,
                is_3bet_opp=is_3bet_opp,
                is_3bet=is_3bet,
                is_fold_to_3bet_opp=is_fold_to_3bet_opp,
                is_fold_to_3bet=is_fold_to_3bet,
                is_cbet_opp=is_cbet_opp,
                is_cbet=is_cbet,
                is_fold_to_cbet_opp=is_fold_to_cbet_opp,
                is_fold_to_cbet=is_fold_to_cbet,
            ))

        # Apply action to state
        stacks[p_name] -= added
        street_bets[p_name] += added
        total_contrib[p_name] += added
        pot += added

        if street == "preflop" and raw_action in ("bet", "raise"):
            preflop_raises += 1
            preflop_aggressor = p_name
            if preflop_raises == 1 and p_name == detected_hero:
                hero_opened_pf = True
        elif street == "flop" and raw_action == "bet" and p_name == preflop_aggressor:
            cbet_made_on_flop = True
            cbet_bettor = p_name

    if hero_hole is None:
        return None

    net_chips = collected[detected_hero] - total_contrib[detected_hero]
    net_bb = net_chips / max(1, bb)

    return ParsedHand(
        hand_id=hand_id,
        source_format=source_format,
        hero_name=detected_hero,
        hero_hole=hero_hole,
        position=hero_pos,
        board=tuple(board),
        big_blind=bb,
        net_chips=net_chips,
        net_bb=net_bb,
        went_to_showdown=went_to_showdown,
        won_at_showdown=won_at_showdown,
        decisions=decisions,
    )


def parse_hand_histories(raw_text: str, hero_name: str | None = None) -> dict:
    """Parse JSON or raw text hand histories (Global Poker, PokerStars, GGPoker)."""
    stripped = raw_text.strip()
    errors: list[str] = []
    hands: list[ParsedHand] = []

    if not stripped:
        return {"hands": [], "hero_name": hero_name or "Hero", "source_format": "unknown", "parsed_count": 0, "errors": ["Empty input"]}

    if stripped.startswith("{") or stripped.startswith("["):
        try:
            data = json.loads(stripped)
            hands = _parse_json_hands(data, hero_name)
            detected = hands[0].hero_name if hands else (hero_name or "Hero")
            return {
                "hands": hands,
                "hero_name": detected,
                "source_format": "json",
                "parsed_count": len(hands),
                "errors": errors,
            }
        except Exception as exc:
            errors.append(f"JSON parse error: {exc}")
            return {"hands": [], "hero_name": hero_name or "Hero", "source_format": "json", "parsed_count": 0, "errors": errors}

    # Split into hand blocks on blank lines or Hand # headers
    blocks = re.split(r"\n\s*\n+(?=(?:Global Poker|PokerStars|GGPoker|Poker Hand|Hand #))", stripped)
    if len(blocks) == 1:
        blocks = re.split(r"(?m)^(?=(?:Global Poker Hand #|PokerStars (?:Hand|Game) #|GGPoker Hand #|Poker Hand #))", stripped)

    for idx, blk in enumerate(blocks):
        blk = blk.strip()
        if not blk:
            continue
        try:
            parsed = _parse_text_hand(blk, hero_name)
            if parsed is not None:
                hands.append(parsed)
        except Exception as exc:
            errors.append(f"Block {idx + 1}: {exc}")

    detected_hero = hands[0].hero_name if hands else (hero_name or "Hero")
    source_format = hands[0].source_format if hands else _detect_format(stripped)
    return {
        "hands": hands,
        "hero_name": detected_hero,
        "source_format": source_format,
        "parsed_count": len(hands),
        "errors": errors,
    }


def compute_player_stats(hands: list[ParsedHand]) -> dict:
    """Compute aggregate HUD, positional, street, and 169-matrix stats."""
    n_hands = len(hands)
    decisions_count = sum(len(h.decisions) for h in hands)

    vpip_hands = 0
    pfr_hands = 0
    three_bet_opps = 0
    three_bets = 0
    fold_to_3bet_opps = 0
    fold_to_3bets = 0
    cbet_opps = 0
    cbets = 0
    fold_to_cbet_opps = 0
    fold_to_cbets = 0
    postflop_aggr = 0
    postflop_calls = 0
    flops_seen = 0
    showdowns = 0
    showdowns_won = 0
    total_net_bb = 0.0

    pos_acc = {pos: {"hands": 0, "vpip": 0, "pfr": 0, "net_bb": 0.0} for pos in POSITIONS}
    street_dist: dict[str, dict[str, int]] = {
        st: {act: 0 for act in ACTION_SPACE} for st in STREETS
    }
    matrix: dict[str, dict] = {
        code: {"hands": 0, "vpip": 0, "pfr": 0, "net_bb": 0.0} for code in _ALL_169_HANDS
    }

    for h in hands:
        total_net_bb += h.net_bb
        if len(h.board) >= 3:
            flops_seen += 1
        if h.went_to_showdown:
            showdowns += 1
            if h.won_at_showdown:
                showdowns_won += 1

        h_vpip = any(d.is_vpip for d in h.decisions)
        h_pfr = any(d.is_pfr for d in h.decisions)
        if h_vpip:
            vpip_hands += 1
        if h_pfr:
            pfr_hands += 1

        pos = h.position if h.position in pos_acc else "BTN"
        pos_acc[pos]["hands"] += 1
        pos_acc[pos]["vpip"] += 1 if h_vpip else 0
        pos_acc[pos]["pfr"] += 1 if h_pfr else 0
        pos_acc[pos]["net_bb"] += h.net_bb

        code = canonical_preflop_hand(h.hero_hole)
        if code in matrix:
            matrix[code]["hands"] += 1
            matrix[code]["vpip"] += 1 if h_vpip else 0
            matrix[code]["pfr"] += 1 if h_pfr else 0
            matrix[code]["net_bb"] += h.net_bb

        for d in h.decisions:
            if d.is_3bet_opp:
                three_bet_opps += 1
            if d.is_3bet:
                three_bets += 1
            if d.is_fold_to_3bet_opp:
                fold_to_3bet_opps += 1
            if d.is_fold_to_3bet:
                fold_to_3bets += 1
            if d.is_cbet_opp:
                cbet_opps += 1
            if d.is_cbet:
                cbets += 1
            if d.is_fold_to_cbet_opp:
                fold_to_cbet_opps += 1
            if d.is_fold_to_cbet:
                fold_to_cbets += 1

            if d.street != "preflop":
                if d.raw_action in ("bet", "raise"):
                    postflop_aggr += 1
                elif d.raw_action == "call":
                    postflop_calls += 1

            if d.street in street_dist and d.action in street_dist[d.street]:
                street_dist[d.street][d.action] += 1

    def pct(num: int, den: int) -> float:
        return round(100.0 * num / den, 1) if den > 0 else 0.0

    position_stats = {}
    for pos, acc in pos_acc.items():
        cnt = acc["hands"]
        position_stats[pos] = {
            "hands": cnt,
            "vpip": pct(acc["vpip"], cnt),
            "pfr": pct(acc["pfr"], cnt),
            "bb_per_100": round(acc["net_bb"] / cnt * 100.0, 1) if cnt > 0 else 0.0,
        }

    preflop_matrix = {}
    for code, acc in matrix.items():
        cnt = acc["hands"]
        preflop_matrix[code] = {
            "hands": cnt,
            "vpip": pct(acc["vpip"], cnt),
            "pfr": pct(acc["pfr"], cnt),
            "net_bb": round(acc["net_bb"], 2),
        }

    return {
        "hands_count": n_hands,
        "decisions_count": decisions_count,
        "vpip": pct(vpip_hands, n_hands),
        "pfr": pct(pfr_hands, n_hands),
        "three_bet": pct(three_bets, three_bet_opps),
        "fold_to_3bet": pct(fold_to_3bets, fold_to_3bet_opps),
        "cbet_flop": pct(cbets, cbet_opps),
        "fold_to_cbet": pct(fold_to_cbets, fold_to_cbet_opps),
        "af": round(postflop_aggr / max(1, postflop_calls), 2) if (postflop_aggr + postflop_calls) > 0 else 0.0,
        "wtsd": pct(showdowns, flops_seen),
        "wsd": pct(showdowns_won, showdowns),
        "bb_per_100": round(total_net_bb / n_hands * 100.0, 1) if n_hands > 0 else 0.0,
        "position_stats": position_stats,
        "street_action_dist": street_dist,
        "preflop_matrix": preflop_matrix,
    }


def parsed_hand_to_dict(hand: ParsedHand) -> dict:
    """Serialize a ParsedHand to a JSON-compatible dict."""
    return asdict(hand)
