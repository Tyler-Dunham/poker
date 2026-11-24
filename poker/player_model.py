"""Hierarchical empirical Bayes + L2-regularized softmax player clone model."""

from __future__ import annotations

import math
from random import Random

from .abstraction import (
    ACTION_SPACE,
    STREETS,
    abstract_to_legal_action,
    encode_observation,
    legal_abstract_actions,
)
from .parser import ParsedHand
from .simulator import Action, Bot, Observation

_NUM_ACTIONS = len(ACTION_SPACE)
_ACT_TO_IDX = {a: i for i, a in enumerate(ACTION_SPACE)}
_NUM_FEATURES = 13


def _softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps] if s > 0 else [1.0 / len(logits)] * len(logits)


class PlayerCloneModel:
    """Clones a player's decision policy using hierarchical EB counts + softmax SGD."""

    def __init__(self) -> None:
        self.state_counts: dict[str, list[float]] = {}
        self.coarse_counts: dict[str, list[float]] = {}
        self.fallback_counts: dict[str, list[float]] = {}
        self.street_counts: dict[str, list[float]] = {
            st: [1.0] * _NUM_ACTIONS for st in STREETS
        }
        # Weight matrix: _NUM_ACTIONS x _NUM_FEATURES
        self.weights: list[list[float]] = [[0.0] * _NUM_FEATURES for _ in range(_NUM_ACTIONS)]
        self.fitted_decisions: int = 0
        self.fitted_hands: int = 0

    def fit(self, hands: list[ParsedHand], epochs: int = 25, lr: float = 0.08, l2: float = 0.01) -> PlayerCloneModel:
        """Fit hierarchical count tables and L2-regularized multinomial softmax via SGD."""
        self.state_counts.clear()
        self.coarse_counts.clear()
        self.fallback_counts.clear()
        self.street_counts = {st: [1.0] * _NUM_ACTIONS for st in STREETS}
        self.weights = [[0.0] * _NUM_FEATURES for _ in range(_NUM_ACTIONS)]

        samples: list[tuple[list[float], int]] = []
        for h in hands:
            for d in h.decisions:
                idx = _ACT_TO_IDX.get(d.action)
                if idx is None:
                    continue
                for table, key in (
                    (self.state_counts, d.state_key),
                    (self.coarse_counts, d.coarse_key),
                    (self.fallback_counts, d.fallback_key),
                ):
                    if key not in table:
                        table[key] = [0.0] * _NUM_ACTIONS
                    table[key][idx] += 1.0
                if d.street in self.street_counts:
                    self.street_counts[d.street][idx] += 1.0
                if len(d.features) == _NUM_FEATURES:
                    samples.append((d.features, idx))

        self.fitted_hands = len(hands)
        self.fitted_decisions = len(samples)

        if not samples:
            return self

        rng = Random(42)
        indices = list(range(len(samples)))
        for epoch in range(epochs):
            rng.shuffle(indices)
            step_lr = lr / (1.0 + 0.05 * epoch)
            for i in indices:
                feats, target = samples[i]
                logits = [
                    sum(w[j] * feats[j] for j in range(_NUM_FEATURES))
                    for w in self.weights
                ]
                probs = _softmax(logits)
                for a in range(_NUM_ACTIONS):
                    grad = probs[a] - (1.0 if a == target else 0.0)
                    w_row = self.weights[a]
                    for j in range(_NUM_FEATURES):
                        w_row[j] -= step_lr * (grad * feats[j] + l2 * w_row[j])

        return self

    def _softmax_probs(self, features: list[float]) -> list[float]:
        if len(features) != _NUM_FEATURES:
            return [1.0 / _NUM_ACTIONS] * _NUM_ACTIONS
        logits = [
            sum(w[j] * features[j] for j in range(_NUM_FEATURES))
            for w in self.weights
        ]
        return _softmax(logits)

    def action_distribution(self, obs_or_encoded: Observation | dict) -> dict[str, float]:
        """Return probability distribution over ACTION_SPACE (or legal actions if Observation)."""
        is_obs = isinstance(obs_or_encoded, Observation)
        encoded = encode_observation(obs_or_encoded) if is_obs else obs_or_encoded

        street = encoded.get("street", "preflop")
        s_key = encoded.get("state_key", "")
        c_key = encoded.get("coarse_key", "")
        f_key = encoded.get("fallback_key", "")
        feats = encoded.get("features", [])

        # Hierarchical Empirical Bayes smoothing
        st_c = self.street_counts.get(street, [1.0] * _NUM_ACTIONS)
        st_tot = sum(st_c)
        p_street = [c / st_tot for c in st_c]

        fb_c = self.fallback_counts.get(f_key)
        if fb_c:
            alpha_fb = 4.0
            p_fb = [(fb_c[i] + alpha_fb * p_street[i]) / (sum(fb_c) + alpha_fb) for i in range(_NUM_ACTIONS)]
        else:
            p_fb = p_street

        c_c = self.coarse_counts.get(c_key)
        if c_c:
            alpha_c = 4.0
            p_coarse = [(c_c[i] + alpha_c * p_fb[i]) / (sum(c_c) + alpha_c) for i in range(_NUM_ACTIONS)]
        else:
            p_coarse = p_fb

        s_c = self.state_counts.get(s_key)
        if s_c:
            alpha_s = 3.0
            n_s = sum(s_c)
            p_eb = [(s_c[i] + alpha_s * p_coarse[i]) / (n_s + alpha_s) for i in range(_NUM_ACTIONS)]
        else:
            n_s = 0.0
            p_eb = p_coarse

        p_lr = self._softmax_probs(feats)
        # Blend EB counts with continuous softmax policy
        eb_weight = min(0.85, 0.35 + 0.50 * (n_s / (n_s + 5.0))) if self.fitted_decisions > 0 else 0.2
        blended = [
            eb_weight * p_eb[i] + (1.0 - eb_weight) * p_lr[i]
            for i in range(_NUM_ACTIONS)
        ]

        dist = {ACTION_SPACE[i]: blended[i] for i in range(_NUM_ACTIONS)}

        if is_obs:
            legal = legal_abstract_actions(obs_or_encoded)
            filtered = {a: dist.get(a, 0.0) for a in legal}
            tot = sum(filtered.values())
            if tot <= 0:
                return {a: 1.0 / len(legal) for a in legal}
            return {a: v / tot for a, v in filtered.items()}

        tot = sum(dist.values())
        return {a: v / tot for a, v in dist.items()} if tot > 0 else {a: 1.0 / _NUM_ACTIONS for a in ACTION_SPACE}

    def to_dict(self) -> dict:
        """Serialize model parameters to a JSON-compatible dictionary."""
        return {
            "state_counts": self.state_counts,
            "coarse_counts": self.coarse_counts,
            "fallback_counts": self.fallback_counts,
            "street_counts": self.street_counts,
            "weights": self.weights,
            "fitted_decisions": self.fitted_decisions,
            "fitted_hands": self.fitted_hands,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PlayerCloneModel:
        """Restore a PlayerCloneModel from a serialized dictionary."""
        model = cls()
        model.state_counts = {k: list(v) for k, v in data.get("state_counts", {}).items()}
        model.coarse_counts = {k: list(v) for k, v in data.get("coarse_counts", {}).items()}
        model.fallback_counts = {k: list(v) for k, v in data.get("fallback_counts", {}).items()}
        model.street_counts = {k: list(v) for k, v in data.get("street_counts", {}).items()}
        if "weights" in data and data["weights"]:
            model.weights = [list(row) for row in data["weights"]]
        model.fitted_decisions = int(data.get("fitted_decisions", 0))
        model.fitted_hands = int(data.get("fitted_hands", 0))
        return model

    @staticmethod
    def classify_archetype(stats: dict) -> dict:
        """Classify player stats into a named poker archetype with tagline and traits."""
        vpip = float(stats.get("vpip", 0.0))
        pfr = float(stats.get("pfr", 0.0))
        fold_3b = float(stats.get("fold_to_3bet", 0.0))
        fold_cbet = float(stats.get("fold_to_cbet", 0.0))
        cbet = float(stats.get("cbet_flop", 0.0))
        three_bet = float(stats.get("three_bet", 0.0))
        af = float(stats.get("af", 0.0))

        traits: list[str] = []
        if vpip >= 32.0 and pfr <= 14.0 and af <= 1.5:
            archetype = "Calling Station"
            tagline = "Plays too many hands passively and refuses to fold postflop."
            traits = [
                f"High VPIP ({vpip:.1f}%) with wide cold-calling range",
                f"Low PFR ({pfr:.1f}%) and passive postflop AF ({af:.2f})",
                f"Low fold-to-cbet ({fold_cbet:.1f}%) — rarely bluffable",
            ]
        elif vpip <= 18.0 and (fold_3b >= 60.0 or fold_cbet >= 60.0):
            archetype = "Over-Folding Nit"
            tagline = "Waits for premiums and surrenders too much equity to aggression."
            traits = [
                f"Tight preflop VPIP ({vpip:.1f}%)",
                f"High fold-to-3bet ({fold_3b:.1f}%) — vulnerable to light 3-bets",
                f"High fold-to-cbet ({fold_cbet:.1f}%) — over-folds flops",
            ]
        elif vpip >= 22.0 and pfr >= 18.0 and (three_bet >= 10.0 or cbet >= 70.0 or af >= 3.0):
            archetype = "Aggro Grinder"
            tagline = "Relentless preflop and postflop pressure; attacks capped ranges."
            traits = [
                f"Active LAG profile ({vpip:.1f}% VPIP / {pfr:.1f}% PFR)",
                f"High 3-bet ({three_bet:.1f}%) and flop c-bet ({cbet:.1f}%)",
                f"High postflop aggression factor ({af:.2f})",
            ]
        elif 19.0 <= vpip <= 27.0 and 15.0 <= pfr <= 23.0:
            archetype = "TAG Regular"
            tagline = "Disciplined hand selection with solid value-heavy aggression."
            traits = [
                f"Balanced VPIP/PFR ({vpip:.1f}% / {pfr:.1f}%)",
                f"Controlled postflop aggression (AF {af:.2f})",
                f"Flop c-bet {cbet:.1f}%",
            ]
        else:
            archetype = "Loose-Aggressive" if vpip > 27.0 else "Tight-Passive"
            tagline = "Unconventional frequencies create exploitable range imbalances."
            traits = [
                f"VPIP {vpip:.1f}% / PFR {pfr:.1f}%",
                f"Fold to c-bet {fold_cbet:.1f}%",
                f"Aggression factor {af:.2f}",
            ]

        return {"archetype": archetype, "tagline": tagline, "traits": traits}


class PlayerCloneBot(Bot):
    """Simulator Bot that samples actions from a fitted PlayerCloneModel."""

    def __init__(self, model: PlayerCloneModel, seed: int | None = None) -> None:
        self.model = model
        self.rng = Random(seed)

    def act(self, observation: Observation) -> Action:
        dist = self.model.action_distribution(observation)
        actions = list(dist.keys())
        weights = [dist[a] for a in actions]
        chosen = self.rng.choices(actions, weights=weights, k=1)[0]
        return abstract_to_legal_action(chosen, observation)
