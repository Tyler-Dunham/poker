"""Stdlib-only ThreadingHTTPServer and REST API for Poker Mirror & RL Exploiter."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import threading
import time
from typing import Any
from urllib.parse import urlparse
import uuid


STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "poker_mirror.db"

MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}

OPPONENT_MODE_MAP = {
    "clone": "self_clone",
    "self_clone": "self_clone",
    "rl": "rl_exploiter",
    "exploiter": "rl_exploiter",
    "rl_exploiter": "rl_exploiter",
    "gto": "gto",
    "mixed": "mixed_table",
    "mixed_table": "mixed_table",
}


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses, tuples, and custom objects to JSON-safe types."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(item) for item in obj]
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return _to_jsonable(obj.to_dict())
    if hasattr(obj, "__dict__"):
        return _to_jsonable(
            {k: v for k, v in vars(obj).items() if not k.startswith("_")}
        )
    return str(obj)


def _dict_to_parsed_hand(d: dict[str, Any]) -> Any:
    """Reconstruct a ParsedHand dataclass from a stored hand dict."""
    from poker.parser import ParsedDecision, ParsedHand

    decisions = [ParsedDecision(**dec) for dec in d.get("decisions", [])]
    return ParsedHand(
        hand_id=str(d.get("hand_id", "")),
        source_format=str(d.get("source_format", "unknown")),
        hero_name=str(d.get("hero_name", "Hero")),
        hero_hole=tuple(d.get("hero_hole", ())),
        position=str(d.get("position", "BTN")),
        board=tuple(d.get("board", ())),
        big_blind=int(d.get("big_blind", 10)),
        net_chips=int(d.get("net_chips", 0)),
        net_bb=float(d.get("net_bb", 0.0)),
        went_to_showdown=bool(d.get("went_to_showdown", False)),
        won_at_showdown=bool(d.get("won_at_showdown", False)),
        decisions=decisions,
    )


def _format_session_state(session: Any, raw_state: dict[str, Any]) -> dict[str, Any]:
    """Enrich InteractiveHandSession.get_state() with UI-friendly seats and raise bounds."""
    num_players = int(raw_state.get("num_players", 2))
    stacks = raw_state.get("stacks", [])
    bets = raw_state.get("bets", [])
    folded = raw_state.get("folded", [])
    seat_names = raw_state.get("seat_names", [])
    button = int(raw_state.get("button", 0))
    hero_hole = raw_state.get("hero_hole", [])
    last_result = raw_state.get("last_result") or {}
    showdown = last_result.get("showdown") or {}
    awaiting_hero = bool(raw_state.get("awaiting_hero", False))
    hand_over = bool(raw_state.get("hand_over", False))

    seats = []
    for idx in range(num_players):
        if idx == 0:
            hole = hero_hole
        elif hand_over and str(idx) in showdown:
            hole = showdown[str(idx)]
        elif not (folded[idx] if idx < len(folded) else False):
            hole = ["??", "??"]
        else:
            hole = []

        seats.append({
            "seat": idx,
            "name": seat_names[idx] if idx < len(seat_names) else f"Seat {idx}",
            "stack": stacks[idx] if idx < len(stacks) else 0,
            "bet": bets[idx] if idx < len(bets) else 0,
            "folded": bool(folded[idx]) if idx < len(folded) else False,
            "is_button": idx == button,
            "is_turn": idx == 0 and awaiting_hero and not hand_over,
            "hole": hole,
        })

    obs = getattr(session, "current_obs", None)
    enriched = dict(raw_state)
    enriched["seats"] = seats
    enriched["hand_number"] = getattr(session, "hand_counter", 1)
    enriched["min_raise_to"] = getattr(obs, "min_raise_to", None) if obs else None
    enriched["max_raise_to"] = getattr(obs, "max_raise_to", None) if obs else None
    enriched["history"] = [
        {
            "street": ev.street,
            "seat": ev.seat,
            "player": seat_names[ev.seat] if ev.seat < len(seat_names) else f"Seat {ev.seat}",
            "kind": ev.kind,
            "amount": ev.amount,
        }
        for ev in getattr(session, "events", [])
    ]
    return enriched


class ApiError(Exception):
    """HTTP-mapped application error."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class PokerService:
    """Coordinates storage, hand parsing, clone modeling, RL training, and live sessions."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        resolved = Path(db_path or os.environ.get("POKER_DB_PATH") or DEFAULT_DB_PATH)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = resolved
        self._lock = threading.Lock()
        self._storage: Any = None
        self._models: dict[int, Any] = {}
        self._sessions: dict[str, dict[str, Any]] = {}

    def _get_storage(self) -> Any:
        if self._storage is None:
            from poker.storage import PokerStorage

            self._storage = PokerStorage(str(self.db_path))
        return self._storage

    def list_samples(self) -> list[dict[str, Any]]:
        from poker.sample_hands import SAMPLE_PACKS

        if isinstance(SAMPLE_PACKS, dict):
            items = []
            for pack_id, info in SAMPLE_PACKS.items():
                entry = dict(_to_jsonable(info)) if isinstance(info, dict) else {"name": str(info)}
                entry.setdefault("id", pack_id)
                items.append(entry)
            return items
        return [_to_jsonable(item) for item in SAMPLE_PACKS]

    def list_profiles(self) -> list[dict[str, Any]]:
        storage = self._get_storage()
        profiles = storage.list_profiles()
        out = []
        for p in profiles:
            item = dict(p)
            stats = item.get("stats") or {}
            item["hands_count"] = stats.get("hands_count", item.get("hands_stored", 0))
            out.append(item)
        return out

    def _fit_clone_model(self, hands: list[Any]) -> Any:
        from poker.player_model import PlayerCloneModel

        model = PlayerCloneModel()
        if hands:
            model.fit(hands)
        return model

    def _extract_stats_payload(self, stats: dict[str, Any], model: Any, archetype: Any = None) -> dict[str, Any]:
        matrix = stats.get("preflop_matrix") or stats.get("matrix_13x13")
        if archetype is None and model is not None and hasattr(model, "classify_archetype"):
            archetype = _to_jsonable(model.classify_archetype(stats))
        return {
            "stats": stats,
            "matrix": matrix,
            "archetype": archetype,
        }

    def import_profile(
        self,
        *,
        name: str,
        raw_text: str,
        hero_name: str | None = None,
        profile_id: str | int | None = None,
    ) -> dict[str, Any]:
        if not raw_text or not raw_text.strip():
            raise ApiError(HTTPStatus.BAD_REQUEST, "raw_text is required")

        from poker.parser import compute_player_stats, parse_hand_histories

        parsed = parse_hand_histories(raw_text, hero_name=hero_name)
        hands = parsed.get("hands", []) if isinstance(parsed, dict) else parsed
        if not hands:
            raise ApiError(HTTPStatus.BAD_REQUEST, "No valid hands could be parsed from raw_text")

        storage = self._get_storage()
        if profile_id is not None and str(profile_id).strip():
            pid = int(profile_id)
            existing = storage.get_profile(pid)
            if not existing:
                raise ApiError(HTTPStatus.NOT_FOUND, f"Profile '{pid}' not found")
            storage.add_hands(pid, hands)
            stored_dicts = storage.get_profile_hands(pid)
            all_hands = [_dict_to_parsed_hand(hd) for hd in stored_dicts]
        else:
            created = storage.create_profile(name=name or "Imported Player")
            pid = int(created["id"])
            storage.add_hands(pid, hands)
            all_hands = hands

        stats = compute_player_stats(all_hands)
        model = self._fit_clone_model(all_hands)
        archetype = model.classify_archetype(stats)
        updated = storage.save_training_result(
            pid,
            stats=stats,
            clone_model=model.to_dict(),
            archetype=archetype,
        )
        with self._lock:
            self._models[pid] = model

        result = dict(updated or {})
        result.update(self._extract_stats_payload(stats, model, archetype=archetype))
        return result

    def create_sample_profile(self, *, pack_id: str, name: str | None = None) -> dict[str, Any]:
        if not pack_id:
            raise ApiError(HTTPStatus.BAD_REQUEST, "pack_id is required")

        from poker.sample_hands import SAMPLE_PACKS, generate_sample_hand_history

        if pack_id not in SAMPLE_PACKS:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"Unknown sample pack '{pack_id}'")

        raw_text = generate_sample_hand_history(pack_id, hero_name="Hero")
        info = SAMPLE_PACKS[pack_id]
        pack_name = name or (info.get("name", pack_id) if isinstance(info, dict) else str(info))
        return self.import_profile(name=pack_name, raw_text=raw_text, hero_name="Hero")

    def _get_or_fit_model(self, profile_id: int, profile_dict: dict[str, Any]) -> Any:
        with self._lock:
            cached = self._models.get(profile_id)
        if cached is not None:
            return cached

        from poker.player_model import PlayerCloneModel

        clone_data = profile_dict.get("clone_model")
        if isinstance(clone_data, dict) and clone_data.get("weights"):
            model = PlayerCloneModel.from_dict(clone_data)
            with self._lock:
                self._models[profile_id] = model
            return model

        storage = self._get_storage()
        stored_dicts = storage.get_profile_hands(profile_id)
        if not stored_dicts:
            return None
        hands = [_dict_to_parsed_hand(hd) for hd in stored_dicts]
        model = self._fit_clone_model(hands)
        with self._lock:
            self._models[profile_id] = model
        return model

    def get_profile(self, profile_id: str) -> dict[str, Any]:
        try:
            pid = int(profile_id)
        except ValueError as exc:
            raise ApiError(HTTPStatus.NOT_FOUND, f"Profile '{profile_id}' not found") from exc

        storage = self._get_storage()
        profile = storage.get_profile(pid)
        if not profile:
            raise ApiError(HTTPStatus.NOT_FOUND, f"Profile '{profile_id}' not found")

        model = self._get_or_fit_model(pid, profile)
        stats = profile.get("stats") or {}
        archetype = profile.get("archetype") or None
        extracted = self._extract_stats_payload(stats, model, archetype=archetype)

        result = dict(profile)
        result.update(extracted)
        exploit_report = profile.get("exploit_report") or {}
        result["rl_report"] = exploit_report if exploit_report else None
        return result

    def train_rl_exploiter(self, profile_id: str, episodes: int = 2500) -> dict[str, Any]:
        try:
            pid = int(profile_id)
        except ValueError as exc:
            raise ApiError(HTTPStatus.NOT_FOUND, f"Profile '{profile_id}' not found") from exc

        storage = self._get_storage()
        profile = storage.get_profile(pid)
        if not profile:
            raise ApiError(HTTPStatus.NOT_FOUND, f"Profile '{profile_id}' not found")

        model = self._get_or_fit_model(pid, profile)
        if model is None:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Profile has no hands to train against")

        from poker.rl_exploiter import GTOBaselineBot, RLExploiterBot, analyze_exploits

        episodes = max(100, min(int(episodes or 2500), 20000))
        t0 = time.perf_counter()

        exploiter = RLExploiterBot()
        train_summary = exploiter.train_against_clone(model, episodes=episodes)
        gto_bot = GTOBaselineBot()
        stats = profile.get("stats") or {}
        report = analyze_exploits(
            stats=stats,
            clone_model=model,
            exploiter=exploiter,
            gto=gto_bot,
        )
        training_ms = round((time.perf_counter() - t0) * 1000, 1)

        report_dict = dict(report)
        report_dict["learning_curve"] = train_summary.get("learning_curve", [])
        report_dict["training_ms"] = training_ms

        storage.save_training_result(
            pid,
            exploiter=exploiter.to_dict(),
            exploit_report=report_dict,
        )
        return report_dict

    def start_game(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw_pid = payload.get("profile_id")
        pid = int(raw_pid) if raw_pid not in (None, "") else None
        raw_mode = str(payload.get("opponent_mode") or "clone")
        mode = OPPONENT_MODE_MAP.get(raw_mode, "self_clone")
        num_seats = int(payload.get("num_seats") or 2)
        starting_stack = int(payload.get("starting_stack") or 1000)
        big_blind = int(payload.get("big_blind") or 10)
        small_blind = max(1, big_blind // 2)

        storage = self._get_storage()
        model = None
        exploiter_bot = None
        if pid is not None:
            profile = storage.get_profile(pid)
            if not profile:
                raise ApiError(HTTPStatus.NOT_FOUND, f"Profile '{pid}' not found")
            model = self._get_or_fit_model(pid, profile)
            exploiter_data = profile.get("exploiter") or {}
            if exploiter_data.get("q_table"):
                from poker.rl_exploiter import RLExploiterBot

                exploiter_bot = RLExploiterBot.from_dict(exploiter_data)

        from poker.rl_exploiter import GTOBaselineBot, RLExploiterBot
        from poker.session import InteractiveHandSession

        if mode in ("rl_exploiter", "mixed_table") and exploiter_bot is None and model is not None:
            exploiter_bot = RLExploiterBot()
            exploiter_bot.train_against_clone(model, episodes=400)

        session = InteractiveHandSession(
            mode=mode,
            num_players=num_seats,
            starting_stack=starting_stack,
            small_blind=small_blind,
            big_blind=big_blind,
            hero_name="Hero",
            clone_model=model,
            exploiter_bot=exploiter_bot,
            gto_bot=GTOBaselineBot(),
        )
        raw_state = session.start_hand()
        session_id = str(uuid.uuid4())
        with self._lock:
            self._sessions[session_id] = {
                "session": session,
                "profile_id": pid,
                "saved_count": 0,
            }

        return {
            "session_id": session_id,
            "state": _format_session_state(session, raw_state),
        }

    def game_action(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            entry = self._sessions.get(session_id)
        if not entry:
            raise ApiError(HTTPStatus.NOT_FOUND, f"Game session '{session_id}' not found")

        session = entry["session"]
        kind = str(payload.get("kind") or "")
        amount = payload.get("amount")
        if not kind:
            raise ApiError(HTTPStatus.BAD_REQUEST, "action kind is required")

        from poker.simulator import Action

        try:
            sim_action = Action(kind=kind, amount=int(amount) if amount is not None else 0)
            raw_state = session.submit_action(sim_action)
        except (ValueError, RuntimeError) as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc

        formatted = _format_session_state(session, raw_state)
        coach_list = formatted.get("coach_feedback") or []
        latest_coach = coach_list[-1] if coach_list else None
        if latest_coach and "rl_exploit_note" in latest_coach and "advice" not in latest_coach:
            latest_coach = dict(latest_coach)
            latest_coach["advice"] = latest_coach["rl_exploit_note"]

        hand_over = bool(formatted.get("hand_over", False))
        last_result = formatted.get("last_result")
        hand_summary = None
        if hand_over and last_result:
            net = last_result.get("net") or [0]
            hand_summary = {
                "hero_net": net[0],
                "summary": f"Hero net: {net[0]:+d} chips",
                **last_result,
            }

        return {
            "state": formatted,
            "coach_feedback": latest_coach,
            "hand_over": hand_over,
            "hand_summary": hand_summary,
        }

    def next_hand(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            entry = self._sessions.get(session_id)
        if not entry:
            raise ApiError(HTTPStatus.NOT_FOUND, f"Game session '{session_id}' not found")

        session = entry["session"]
        raw_state = session.start_hand()
        return {"state": _format_session_state(session, raw_state)}

    def save_session_to_profile(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            entry = self._sessions.get(session_id)
        if not entry:
            raise ApiError(HTTPStatus.NOT_FOUND, f"Game session '{session_id}' not found")

        pid = entry.get("profile_id")
        if pid is None:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Session is not attached to a profile")

        session = entry["session"]
        completed = list(getattr(session, "completed_hands", []))
        saved_count = entry["saved_count"]
        new_hand_dicts = completed[saved_count:]

        from poker.parser import compute_player_stats, parse_hand_histories

        storage = self._get_storage()
        profile = storage.get_profile(pid)
        if not profile:
            raise ApiError(HTTPStatus.NOT_FOUND, f"Profile '{pid}' not found")

        hands_added = 0
        if new_hand_dicts:
            parsed = parse_hand_histories(json.dumps({"hands": new_hand_dicts}), hero_name="Hero")
            new_parsed_hands = parsed.get("hands", []) if isinstance(parsed, dict) else parsed
            if new_parsed_hands:
                hands_added = storage.add_hands(pid, new_parsed_hands)
            entry["saved_count"] = len(completed)

        stored_dicts = storage.get_profile_hands(pid)
        all_hands = [_dict_to_parsed_hand(hd) for hd in stored_dicts]
        stats = compute_player_stats(all_hands)
        model = self._fit_clone_model(all_hands)
        archetype = model.classify_archetype(stats)
        storage.save_training_result(
            pid,
            stats=stats,
            clone_model=model.to_dict(),
            archetype=archetype,
        )
        with self._lock:
            self._models[pid] = model

        return {
            "hands_added": hands_added,
            "stats": stats,
            "archetype": archetype,
        }


class PokerHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        RequestHandlerClass: type[BaseHTTPRequestHandler],
        service: PokerService,
    ) -> None:
        self.service = service
        super().__init__(server_address, RequestHandlerClass)


class PokerRequestHandler(BaseHTTPRequestHandler):
    """REST API + static SPA handler."""

    server: PokerHTTPServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        if os.environ.get("POKER_QUIET_LOGS") == "1":
            return
        super().log_message(format, *args)

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(_to_jsonable(payload)).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_path: Path) -> None:
        try:
            resolved = file_path.resolve()
            static_root = STATIC_DIR.resolve()
            if resolved != static_root and static_root not in resolved.parents:
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "Forbidden"})
                return
            if not resolved.is_file():
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return
            data = resolved.read_bytes()
        except OSError as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return

        content_type = MIME_TYPES.get(
            resolved.suffix.lower(),
            mimetypes.guess_type(str(resolved))[0] or "application/octet-stream",
        )
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"Invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "JSON body must be an object")
        return parsed

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/" or path == "/index.html":
                self._send_file(STATIC_DIR / "index.html")
                return
            if path.startswith("/static/"):
                rel = path[len("/static/"):]
                self._send_file(STATIC_DIR / rel)
                return
            if path == "/api/health":
                self._send_json(HTTPStatus.OK, {"ok": True})
                return
            if path == "/api/samples":
                self._send_json(HTTPStatus.OK, self.server.service.list_samples())
                return
            if path == "/api/profiles":
                self._send_json(HTTPStatus.OK, self.server.service.list_profiles())
                return
            m = re.fullmatch(r"/api/profiles/([^/]+)", path)
            if m:
                self._send_json(HTTPStatus.OK, self.server.service.get_profile(m.group(1)))
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {path}"})
        except ApiError as exc:
            self._send_json(exc.status, {"error": exc.message})
        except Exception as exc:  # noqa: BLE001
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            body = self._read_json_body()
            if path == "/api/profiles/import":
                res = self.server.service.import_profile(
                    name=str(body.get("name") or "Imported Player"),
                    hero_name=body.get("hero_name"),
                    raw_text=str(body.get("raw_text") or ""),
                    profile_id=body.get("profile_id"),
                )
                self._send_json(HTTPStatus.OK, res)
                return
            if path == "/api/profiles/sample":
                res = self.server.service.create_sample_profile(
                    pack_id=str(body.get("pack_id") or ""),
                    name=body.get("name"),
                )
                self._send_json(HTTPStatus.OK, res)
                return
            m = re.fullmatch(r"/api/profiles/([^/]+)/train", path)
            if m:
                episodes = int(body.get("episodes") or 2500)
                res = self.server.service.train_rl_exploiter(m.group(1), episodes=episodes)
                self._send_json(HTTPStatus.OK, res)
                return
            if path == "/api/game/start":
                res = self.server.service.start_game(body)
                self._send_json(HTTPStatus.OK, res)
                return
            m = re.fullmatch(r"/api/game/([^/]+)/action", path)
            if m:
                res = self.server.service.game_action(m.group(1), body)
                self._send_json(HTTPStatus.OK, res)
                return
            m = re.fullmatch(r"/api/game/([^/]+)/next_hand", path)
            if m:
                res = self.server.service.next_hand(m.group(1))
                self._send_json(HTTPStatus.OK, res)
                return
            m = re.fullmatch(r"/api/game/([^/]+)/save_to_profile", path)
            if m:
                res = self.server.service.save_session_to_profile(m.group(1))
                self._send_json(HTTPStatus.OK, res)
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {path}"})
        except ApiError as exc:
            self._send_json(exc.status, {"error": exc.message})
        except Exception as exc:  # noqa: BLE001
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})


def create_server(
    host: str = "127.0.0.1",
    port: int | None = None,
    db_path: str | Path | None = None,
) -> PokerHTTPServer:
    """Create a configured PokerHTTPServer instance."""
    resolved_port = int(port if port is not None else os.environ.get("PORT", "8081"))
    service = PokerService(db_path=db_path)
    return PokerHTTPServer((host, resolved_port), PokerRequestHandler, service=service)


def main() -> None:
    parser = argparse.ArgumentParser(description="Poker Mirror & RL Exploiter web server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8081")),
        help="Listen port (default: PORT env var or 8081)",
    )
    parser.add_argument("--db", default=None, help="Path to SQLite database file")
    args = parser.parse_args()

    server = create_server(host=args.host, port=args.port, db_path=args.db)
    actual_host, actual_port = server.server_address[:2]
    print(f"Poker Mirror Studio listening on http://{actual_host}:{actual_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
