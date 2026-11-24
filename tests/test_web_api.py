"""End-to-end HTTP integration tests for the Poker Mirror web server."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from web.server import create_server


class WebApiIntegrationTests(unittest.TestCase):
    """Spin up ThreadingHTTPServer on an ephemeral port and exercise the REST API."""

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["POKER_QUIET_LOGS"] = "1"
        cls._tmp_dir = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls._tmp_dir.name) / "test_poker_mirror.db"
        cls.server = create_server(host="127.0.0.1", port=0, db_path=cls.db_path)
        cls.port = cls.server.server_address[1]
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        cls._tmp_dir.cleanup()

    def _request_json(
        self, method: str, path: str, payload: dict | None = None
    ) -> tuple[int, dict | list]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        req = Request(f"{self.base_url}{path}", data=data, headers=headers, method=method)
        try:
            with urlopen(req, timeout=15) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except HTTPError as err:
            body = json.loads(err.read().decode("utf-8"))
            return err.code, body

    def test_static_and_health(self) -> None:
        with urlopen(f"{self.base_url}/", timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            html = resp.read().decode("utf-8")
            self.assertIn("Poker Mirror", html)

        with urlopen(f"{self.base_url}/static/styles.css", timeout=5) as resp:
            self.assertEqual(resp.status, 200)

        status, body = self._request_json("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"ok": True})

    def test_full_profile_train_and_play_flow(self) -> None:
        # 1. List sample packs
        status, samples = self._request_json("GET", "/api/samples")
        self.assertEqual(status, 200)
        self.assertIsInstance(samples, list)
        self.assertGreater(len(samples), 0)
        pack_id = samples[0]["id"]

        # 2. Create sample profile
        status, profile = self._request_json(
            "POST", "/api/profiles/sample", {"pack_id": pack_id, "name": "Test Clone"}
        )
        self.assertEqual(status, 200)
        self.assertIn("id", profile)
        self.assertIn("stats", profile)
        self.assertIn("matrix", profile)
        self.assertIn("archetype", profile)
        profile_id = profile["id"]

        # 3. Verify GET /api/profiles and GET /api/profiles/{id}
        status, profiles = self._request_json("GET", "/api/profiles")
        self.assertEqual(status, 200)
        self.assertTrue(any(p["id"] == profile_id for p in profiles))

        status, fetched = self._request_json("GET", f"/api/profiles/{profile_id}")
        self.assertEqual(status, 200)
        self.assertEqual(fetched["id"], profile_id)

        # 4. Train RL Exploiter
        status, report = self._request_json(
            "POST", f"/api/profiles/{profile_id}/train", {"episodes": 500}
        )
        self.assertEqual(status, 200)
        for key in (
            "learning_curve",
            "exploitability_bb100",
            "overall_score",
            "exploits",
            "strengths",
            "spot_comparisons",
            "training_ms",
        ):
            self.assertIn(key, report)

        # 5. Start interactive game session
        status, game = self._request_json(
            "POST",
            "/api/game/start",
            {
                "profile_id": profile_id,
                "opponent_mode": "clone",
                "num_seats": 2,
                "starting_stack": 1000,
                "big_blind": 10,
            },
        )
        self.assertEqual(status, 200)
        session_id = game["session_id"]
        state = game["state"]

        # 6. Play until hand_over using legal actions
        hand_over = False
        steps = 0
        while not hand_over and steps < 20:
            legal = state.get("legal_actions") or []
            if "check" in legal:
                action = {"kind": "check"}
            elif "call" in legal:
                action = {"kind": "call"}
            elif "fold" in legal:
                action = {"kind": "fold"}
            else:
                break

            status, res = self._request_json(
                "POST", f"/api/game/{session_id}/action", action
            )
            self.assertEqual(status, 200)
            self.assertIn("state", res)
            self.assertIn("coach_feedback", res)
            state = res["state"]
            hand_over = bool(res.get("hand_over"))
            steps += 1

        self.assertTrue(hand_over, "Expected hand to finish within 20 actions")

        # 7. Save played hand back to profile
        status, saved = self._request_json(
            "POST", f"/api/game/{session_id}/save_to_profile", {}
        )
        self.assertEqual(status, 200)
        self.assertIn("hands_added", saved)
        self.assertGreaterEqual(saved["hands_added"], 1)
        self.assertIn("stats", saved)
        self.assertIn("archetype", saved)


if __name__ == "__main__":
    unittest.main()
