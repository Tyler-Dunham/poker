"""Comprehensive unit tests for the pure-Python RL poker trainer backend."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from poker.abstraction import (
    ACTION_SPACE,
    POSITIONS,
    STREETS,
    abstract_to_legal_action,
    canonical_preflop_hand,
    encode_observation,
    facing_bucket,
    legal_abstract_actions,
    position_name,
    postflop_strength_bucket,
    preflop_strength_bucket,
    spr_bucket,
)
from poker.parser import compute_player_stats, parse_hand_histories
from poker.player_model import PlayerCloneBot, PlayerCloneModel
from poker.rl_exploiter import GTOBaselineBot, RLExploiterBot, analyze_exploits
from poker.sample_hands import SAMPLE_PACKS, generate_sample_hand_history
from poker.session import InteractiveHandSession
from poker.simulator import Action, Observation, PokerGame
from poker.storage import PokerStorage


class AbstractionTests(unittest.TestCase):
    def test_canonical_and_strength_buckets(self):
        self.assertEqual(canonical_preflop_hand(("As", "Ah")), "AA")
        self.assertEqual(canonical_preflop_hand(("Kh", "Ah")), "AKs")
        self.assertEqual(canonical_preflop_hand(("2c", "7d")), "72o")
        self.assertEqual(preflop_strength_bucket(("As", "Ad")), 5)
        self.assertEqual(preflop_strength_bucket(("7c", "2d")), 0)

        b_flush, tex_wet = postflop_strength_bucket(("As", "Ks"), ("Qs", "Js", "2s"))
        self.assertEqual(b_flush, 5)
        self.assertEqual(tex_wet, "wet")

        _, tex_paired = postflop_strength_bucket(("As", "Kd"), ("Ac", "Ad", "7h"))
        self.assertEqual(tex_paired, "paired")

        _, tex_dry = postflop_strength_bucket(("As", "Kd"), ("Kh", "7c", "2s"))
        self.assertEqual(tex_dry, "dry")

    def test_position_facing_and_spr(self):
        self.assertEqual(position_name(0, 0, 2), "BTN")
        self.assertEqual(position_name(1, 0, 2), "BB")
        six_max = [position_name(i, 0, 6) for i in range(6)]
        self.assertEqual(six_max, ["BTN", "SB", "BB", "EP", "MP", "CO"])
        self.assertEqual(facing_bucket(0, 100), "none")
        self.assertEqual(facing_bucket(25, 100), "small")
        self.assertEqual(facing_bucket(60, 100), "medium")
        self.assertEqual(facing_bucket(120, 100), "large")
        self.assertEqual(spr_bucket(200, 100), "short")
        self.assertEqual(spr_bucket(500, 100), "medium")
        self.assertEqual(spr_bucket(1000, 100), "deep")

    def test_legal_action_edge_cases(self):
        # Case 1: to_call == 0, can bet
        obs_open = Observation(
            seat=0, button=0, street="flop", hole=("As", "Kd"), board=("Ac", "7h", "2d"),
            pot=100, stacks=(900, 900), bets=(0, 0), folded=(False, False),
            to_call=0, min_raise_to=10, max_raise_to=900,
            legal_actions=("check", "bet"), history=(),
        )
        self.assertNotIn("fold", legal_abstract_actions(obs_open))
        self.assertEqual(abstract_to_legal_action("fold", obs_open), Action("check"))
        bet_act = abstract_to_legal_action("bet_medium", obs_open)
        self.assertEqual(bet_act.kind, "bet")
        self.assertTrue(10 <= bet_act.amount <= 900)

        # Case 2: min_raise_to is None (cannot raise)
        obs_no_raise = Observation(
            seat=0, button=0, street="turn", hole=("As", "Kd"), board=("Ac", "7h", "2d", "3s"),
            pot=200, stacks=(50, 500), bets=(0, 50), folded=(False, False),
            to_call=50, min_raise_to=None, max_raise_to=None,
            legal_actions=("fold", "call"), history=(),
        )
        self.assertEqual(legal_abstract_actions(obs_no_raise), ["fold", "check_call"])
        self.assertEqual(abstract_to_legal_action("bet_large", obs_no_raise), Action("call"))

        # Case 3: short all-in min_raise_to == max_raise_to
        obs_short = Observation(
            seat=0, button=0, street="river", hole=("As", "Kd"), board=("Ac", "7h", "2d", "3s", "9c"),
            pot=300, stacks=(120, 500), bets=(0, 100), folded=(False, False),
            to_call=100, min_raise_to=120, max_raise_to=120,
            legal_actions=("fold", "call", "raise"), history=(),
        )
        all_in = abstract_to_legal_action("bet_small", obs_short)
        self.assertEqual(all_in, Action("raise", 120))

        enc = encode_observation(obs_open)
        for key in ("state_key", "coarse_key", "fallback_key", "street", "position", "hand_bucket", "hand_code", "facing", "spr", "texture", "features"):
            self.assertIn(key, enc)


class ParserAndStatsTests(unittest.TestCase):
    def test_parse_global_poker_pokerstars_ggpoker_and_json(self):
        gp_text = """
Global Poker Hand #GP1001: Hold'em No Limit ($0.05/$0.10 USD) - 2026/11/20 12:00:00 ET
Table 'Pacific' 6-max Seat #1 is the button
Seat 1: Alice ($10.00 in chips)
Seat 2: Bob ($10.00 in chips)
Seat 3: Carol ($10.00 in chips)
Bob: posts small blind $0.05
Carol: posts big blind $0.10
*** HOLE CARDS ***
Dealt to Alice [A♠ 10s]
Alice: raises $0.20 to $0.30
Bob: folds
Carol: calls $0.20
*** FLOP *** [Ac 7d 2h]
Carol: checks
Alice: bets $0.40
Carol: folds
Uncalled bet ($0.40) returned to Alice
Alice collected $0.65 from pot
*** SUMMARY ***
"""
        ps_text = """
PokerStars Hand #PS2002: Hold'em No Limit ($0.50/$1.00 USD) - 2026/11/20 12:05:00 ET
Table 'Vega' 2-max Seat #1 is the button
Seat 1: Hero ($100.00 in chips)
Seat 2: Villain ($100.00 in chips)
Hero: posts small blind $0.50
Villain: posts big blind $1.00
*** HOLE CARDS ***
Dealt to Hero [Kh Kd]
Hero: raises $1.50 to $2.50
Villain: raises $6.50 to $9.00
Hero: calls $6.50
*** FLOP *** [Qs 8d 3c]
Villain: bets $10.00
Hero: calls $10.00
*** SHOW DOWN ***
Villain: shows [Ac Jd]
Hero: shows [Kh Kd]
Hero collected $38.00 from pot
*** SUMMARY ***
"""
        gg_text = """
Poker Hand #HD3003: Hold'em No Limit ($0.10/$0.25 USD) - 2026/11/20 12:10:00
Table 'GG1' 2-max Seat #1 is the button
Seat 1: Grinder ($25.00 in chips)
Seat 2: Fish ($25.00 in chips)
Grinder: posts small blind $0.10
Fish: posts big blind $0.25
*** HOLE CARDS ***
Dealt to Grinder [Q♥ 10♣]
Grinder: calls $0.15
Fish: checks
*** FLOP *** [Q♠ 5♦ 2♣]
Fish: checks
Grinder: checks
*** SHOW DOWN ***
Fish: shows [9s 4s]
Grinder: shows [Qh Tc]
Grinder collected $0.50 from pot
*** SUMMARY ***
"""
        gp_res = parse_hand_histories(gp_text)
        self.assertEqual(gp_res["parsed_count"], 1)
        self.assertEqual(gp_res["hero_name"], "Alice")
        self.assertEqual(gp_res["source_format"], "global_poker")
        self.assertEqual(gp_res["hands"][0].hero_hole, ("As", "Ts"))

        ps_res = parse_hand_histories(ps_text)
        self.assertEqual(ps_res["parsed_count"], 1)
        self.assertEqual(ps_res["source_format"], "pokerstars")
        self.assertTrue(ps_res["hands"][0].won_at_showdown)

        gg_res = parse_hand_histories(gg_text)
        self.assertEqual(gg_res["parsed_count"], 1)
        self.assertEqual(gg_res["source_format"], "ggpoker")
        self.assertEqual(gg_res["hands"][0].hero_hole, ("Qh", "Tc"))

        # JSON format
        json_payload = json.dumps({
            "hands": [{
                "hand_id": "j1",
                "hero_name": "Hero",
                "hero_hole": ["As", "Kd"],
                "position": "BTN",
                "board": ["Ac", "7h", "2d"],
                "big_blind": 10,
                "net_chips": 40,
                "went_to_showdown": True,
                "won_at_showdown": True,
                "decisions": [
                    {"street": "preflop", "raw_action": "raise", "amount": 30, "pot_before": 15, "to_call": 5, "is_vpip": True, "is_pfr": True},
                    {"street": "flop", "raw_action": "bet", "amount": 40, "pot_before": 60, "to_call": 0, "is_cbet_opp": True, "is_cbet": True},
                ],
            }]
        })
        j_res = parse_hand_histories(json_payload)
        self.assertEqual(j_res["parsed_count"], 1)
        stats = compute_player_stats(j_res["hands"])
        self.assertEqual(stats["hands_count"], 1)
        self.assertEqual(stats["vpip"], 100.0)
        self.assertEqual(stats["cbet_flop"], 100.0)
        self.assertEqual(len(stats["preflop_matrix"]), 169)


class SampleCloneAndRLTests(unittest.TestCase):
    def test_sample_packs_and_clone_model_roundtrip(self):
        self.assertEqual(set(SAMPLE_PACKS.keys()), {"calling_station", "over_folding_nit", "aggro_grinder"})
        raw_cs = generate_sample_hand_history("calling_station", num_hands=180)
        raw_nit = generate_sample_hand_history("over_folding_nit", num_hands=180)
        raw_lag = generate_sample_hand_history("aggro_grinder", num_hands=180)

        st_cs = compute_player_stats(parse_hand_histories(raw_cs)["hands"])
        st_nit = compute_player_stats(parse_hand_histories(raw_nit)["hands"])
        st_lag = compute_player_stats(parse_hand_histories(raw_lag)["hands"])

        self.assertGreater(st_cs["vpip"], st_nit["vpip"] + 15.0)
        self.assertGreater(st_lag["pfr"], st_cs["pfr"] + 8.0)
        self.assertGreater(st_nit["fold_to_cbet"], st_cs["fold_to_cbet"] + 25.0)

        hands_lag = parse_hand_histories(raw_lag)["hands"]
        clone = PlayerCloneModel().fit(hands_lag)
        restored = PlayerCloneModel.from_dict(clone.to_dict())
        obs = Observation(
            seat=0, button=0, street="preflop", hole=("As", "Ks"), board=(),
            pot=15, stacks=(995, 990), bets=(5, 10), folded=(False, False),
            to_call=5, min_raise_to=20, max_raise_to=1000,
            legal_actions=("fold", "call", "raise"), history=(),
        )
        d1 = clone.action_distribution(obs)
        d2 = restored.action_distribution(obs)
        self.assertAlmostEqual(sum(d1.values()), 1.0, places=5)
        for k in d1:
            self.assertAlmostEqual(d1[k], d2[k], places=6)

        arch = PlayerCloneModel.classify_archetype(st_cs)
        self.assertIn("archetype", arch)
        self.assertIn("tagline", arch)
        self.assertIn("traits", arch)

    def test_rl_training_distinct_exploits_and_chip_conservation(self):
        nit_hands = parse_hand_histories(generate_sample_hand_history("over_folding_nit", num_hands=180))["hands"]
        cs_hands = parse_hand_histories(generate_sample_hand_history("calling_station", num_hands=180))["hands"]

        nit_clone = PlayerCloneModel().fit(nit_hands)
        cs_clone = PlayerCloneModel().fit(cs_hands)

        rl_nit = RLExploiterBot(seed=42)
        res_nit = rl_nit.train_against_clone(nit_clone, episodes=2500, seed=42)
        self.assertLess(res_nit["elapsed_seconds"], 4.0)
        self.assertEqual(len(res_nit["learning_curve"]), 10)
        self.assertGreater(res_nit["winrate_bb100"], 0.0)

        rl_cs = RLExploiterBot(seed=42)
        res_cs = rl_cs.train_against_clone(cs_clone, episodes=2500, seed=42)
        self.assertLess(res_cs["elapsed_seconds"], 4.0)
        self.assertGreater(res_cs["winrate_bb100"], 0.0)

        # Verify distinct policies learned for nit vs calling station on a weak flop bluff spot
        bluff_spot = encode_observation(Observation(
            seat=0, button=0, street="flop", hole=("9c", "4d"), board=("Ks", "8d", "2c"),
            pot=60, stacks=(970, 970), bets=(0, 0), folded=(False, False),
            to_call=0, min_raise_to=10, max_raise_to=970,
            legal_actions=("check", "bet"), history=(),
        ))
        dist_nit = rl_nit.action_distribution(bluff_spot)
        dist_cs = rl_cs.action_distribution(bluff_spot)
        # Policies should differ meaningfully
        tvd = 0.5 * sum(abs(dist_nit[a] - dist_cs[a]) for a in ACTION_SPACE)
        self.assertGreater(tvd, 0.05)

        # Serialization roundtrip
        rl_restored = RLExploiterBot.from_dict(rl_nit.to_dict())
        self.assertEqual(rl_restored.training_summary["episodes"], 2500)

        # Chip conservation in PokerGame
        gto = GTOBaselineBot(seed=7)
        clone_bot = PlayerCloneBot(cs_clone, seed=8)
        game = PokerGame([rl_cs, clone_bot, gto], starting_stack=1000, seed=11)
        results = game.run(30)
        self.assertTrue(results)
        self.assertEqual(sum(game.stacks), 3000)
        self.assertTrue(all(sum(r.net) == 0 for r in results))

        # Exploit report
        stats_nit = compute_player_stats(nit_hands)
        report = analyze_exploits(stats_nit, nit_clone, rl_nit, gto)
        for key in ("overall_score", "exploitability_bb100", "exploits", "strengths", "spot_comparisons"):
            self.assertIn(key, report)


class SessionAndStorageTests(unittest.TestCase):
    def test_interactive_session_hu_and_6max_and_json_export(self):
        hands = parse_hand_histories(generate_sample_hand_history("aggro_grinder", num_hands=60))["hands"]
        clone = PlayerCloneModel().fit(hands)
        rl = RLExploiterBot(seed=5)
        rl.train_against_clone(clone, episodes=500, seed=5)

        for mode, n_players in (("self_clone", 2), ("rl_exploiter", 2), ("gto", 6), ("mixed_table", 6)):
            sess = InteractiveHandSession(
                mode=mode, num_players=n_players, clone_model=clone, exploiter_bot=rl, seed=99
            )
            state = sess.start_hand()
            steps = 0
            while not state["hand_over"] and steps < 20:
                self.assertTrue(state["awaiting_hero"])
                legal_abs = state["legal_abstract_actions"]
                chosen = "check_call" if "check_call" in legal_abs else legal_abs[0]
                state = sess.submit_action(chosen)
                steps += 1
            self.assertTrue(state["hand_over"])
            self.assertEqual(sum(state["stacks"]), n_players * 1000)
            if state["coach_feedback"]:
                fb = state["coach_feedback"][0]
                for k in ("your_action", "clone_probs", "gto_probs", "rl_exploit_note", "match_clone_pct"):
                    self.assertIn(k, fb)

            exported = sess.export_hands_json()
            re_parsed = parse_hand_histories(exported)
            self.assertEqual(re_parsed["parsed_count"], 1)

    def test_sqlite_storage_crud(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_mirror.db"
            store = PokerStorage(db_path=db_path)

            prof = store.create_profile("TestHero", archetype={"archetype": "TAG Regular"})
            self.assertEqual(prof["name"], "TestHero")
            self.assertEqual(len(store.list_profiles()), 1)

            hands = parse_hand_histories(generate_sample_hand_history("calling_station", num_hands=20))["hands"]
            inserted = store.add_hands(prof["id"], hands)
            self.assertEqual(inserted, 20)
            stored_hands = store.get_profile_hands(prof["id"])
            self.assertEqual(len(stored_hands), 20)

            updated = store.save_training_result(
                prof["id"],
                stats={"vpip": 40.0},
                clone_model={"fitted_hands": 20},
                exploiter={"episodes": 2500},
                exploit_report={"overall_score": 65.0},
            )
            self.assertIsNotNone(updated)
            self.assertEqual(updated["stats"]["vpip"], 40.0)
            self.assertEqual(store.get_profile(prof["id"])["exploit_report"]["overall_score"], 65.0)

            self.assertTrue(store.delete_profile(prof["id"]))
            self.assertIsNone(store.get_profile(prof["id"]))
            self.assertEqual(store.get_profile_hands(prof["id"]), [])


if __name__ == "__main__":
    unittest.main()
