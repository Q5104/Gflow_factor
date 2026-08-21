import json
import tempfile
import unittest
from pathlib import Path

from factor_gfn.backtest.candidate_import import import_candidate_source_set
from factor_gfn.backtest.expression_compatibility import (
    AUTO_ACCEPT,
    AUTO_REJECT,
    REVIEW_REQUIRED,
    _audit_representation,
    _find_grammar_path,
    _group_status,
    _synthetic_fixture,
    action_registry_for_vocabulary,
    audit_expression_compatibility,
)
from factor_gfn.backtest.sources import CandidateSourceSpec, materialize_source_set
from factor_gfn.evaluator import FactorInterpreter
from factor_gfn.grammar import (
    DAILY_DERIVED_ACTION_REGISTRY,
    Expression,
    get_action_id,
)
from factor_gfn.grammar.tokens import action_space_fingerprint


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


class ExpressionCompatibilityTests(unittest.TestCase):
    def _expression(self) -> Expression:
        return Expression.from_prefix(
            [get_action_id("add"), get_action_id("open"), get_action_id("close")]
        )

    def _representation(self, expression: Expression | None = None) -> dict:
        expression = self._expression() if expression is None else expression
        return {
            "representation_digest": "r" * 64,
            "formula": expression.to_formula(),
            "prefix_token_ids": list(expression.to_prefix()),
            "node_count": expression.stats.node_count,
            "depth": expression.stats.depth,
            "origin_ids": ["origin"],
        }

    def _source_semantics(self, relation: str = "same") -> dict:
        return {
            "source": {
                "action_space_relation": relation,
            }
        }

    def _origin(self) -> list[dict]:
        return [{"provenance": {"source_id": "source"}}]

    def _audit(self, representation, *, claimed_hash=None, relation="same"):
        expression = self._expression()
        result, rebuilt = _audit_representation(
            representation,
            claimed_hash=(
                expression.structural_hash()
                if claimed_hash is None
                else claimed_hash
            ),
            origin_rows=self._origin(),
            source_semantics=self._source_semantics(relation),
            interpreter=FactorInterpreter(_synthetic_fixture()),
            smoke_cache={},
        )
        return result, rebuilt

    def test_exact_current_representation_auto_accepts(self) -> None:
        result, rebuilt = self._audit(self._representation())
        self.assertEqual(result["status"], AUTO_ACCEPT)
        self.assertIsNotNone(rebuilt)
        self.assertTrue(result["grammar"]["reachable"])
        self.assertTrue(result["interpreter_smoke"]["passed"])

    def test_formula_hash_node_and_depth_mismatches_require_review(self) -> None:
        cases = []
        formula = self._representation()
        formula["formula"] = "different"
        cases.append((formula, None, "source_formula_mismatch"))
        node = self._representation()
        node["node_count"] = 99
        cases.append((node, None, "source_node_count_mismatch"))
        depth = self._representation()
        depth["depth"] = 99
        cases.append((depth, None, "source_depth_mismatch"))
        cases.append(
            (self._representation(), "f" * 64, "source_structural_hash_mismatch")
        )
        for representation, claimed_hash, reason in cases:
            with self.subTest(reason=reason):
                result, _ = self._audit(
                    representation, claimed_hash=claimed_hash
                )
                self.assertEqual(result["status"], REVIEW_REQUIRED)
                self.assertIn(reason, result["reason_codes"])

    def test_different_action_space_with_matching_formula_can_accept(self) -> None:
        result, _ = self._audit(self._representation(), relation="different")
        self.assertEqual(result["status"], AUTO_ACCEPT)
        self.assertEqual(
            result["source_action_space_relations"]["source"], "different"
        )

    def test_different_action_space_without_formula_requires_review(self) -> None:
        representation = self._representation()
        representation["formula"] = None
        result, _ = self._audit(representation, relation="different")
        self.assertEqual(result["status"], REVIEW_REQUIRED)
        self.assertIn("action_space_changed_without_formula", result["reason_codes"])

    def test_invalid_prefix_auto_rejects(self) -> None:
        representation = self._representation()
        representation["prefix_token_ids"] = [999]
        result, rebuilt = self._audit(representation)
        self.assertEqual(result["status"], AUTO_REJECT)
        self.assertIn("prefix_parse_invalid", result["reason_codes"])
        self.assertIsNone(rebuilt)

    def test_derived_prefix_round_trip_uses_derived_registry(self) -> None:
        registry = DAILY_DERIVED_ACTION_REGISTRY
        expression = Expression.from_prefix(
            [
                registry.get_action_id("ts_mean", 5),
                registry.get_action_id("ret_cc1"),
            ],
            action_registry=registry,
        )
        vocabulary = {
            "feature_space_id": registry.feature_space.feature_space_id,
            "feature_space_fingerprint": registry.feature_space_fingerprint,
            "action_space_fingerprint": registry.fingerprint(),
        }
        representation = self._representation(expression)
        representation["vocabulary"] = vocabulary
        result, rebuilt = self._audit(
            representation,
            claimed_hash=expression.structural_hash(),
        )
        self.assertEqual(result["status"], AUTO_ACCEPT)
        self.assertEqual(result["vocabulary"], vocabulary)
        self.assertIsNotNone(rebuilt)
        assert rebuilt is not None
        self.assertEqual(rebuilt.action_registry, registry)
        self.assertEqual(rebuilt.to_formula(), expression.to_formula())

    def test_unknown_vocabulary_fails_closed(self) -> None:
        vocabulary = {
            "feature_space_id": "daily_derived_v1",
            "feature_space_fingerprint": "0" * 64,
            "action_space_fingerprint": "1" * 64,
        }
        representation = self._representation()
        representation["vocabulary"] = vocabulary
        result, rebuilt = self._audit(representation)
        self.assertEqual(result["status"], AUTO_REJECT)
        self.assertIn("vocabulary_invalid", result["reason_codes"])
        self.assertIsNone(rebuilt)
        with self.assertRaisesRegex(ValueError, "未知|不匹配"):
            action_registry_for_vocabulary(vocabulary)

    def test_grammar_path_is_early_exit_and_respects_depth_boundary(self) -> None:
        expression = self._expression()
        path = _find_grammar_path(expression)
        self.assertIsNotNone(path)
        self.assertEqual(len(path), expression.stats.node_count)
        prefix = [get_action_id("neg")] * 7 + [get_action_id("open")]
        too_deep = Expression.from_prefix(prefix)
        self.assertIsNone(_find_grammar_path(too_deep))

    def test_all_nan_smoke_result_is_allowed(self) -> None:
        expression = Expression.from_prefix(
            [
                get_action_id("ts_mean", 60),
                get_action_id("ts_mean", 60),
                get_action_id("open"),
            ]
        )
        result, _ = self._audit(self._representation(expression), claimed_hash=expression.structural_hash())
        self.assertEqual(result["status"], AUTO_ACCEPT)
        self.assertEqual(result["interpreter_smoke"]["finite_count"], 0)

    def test_representation_conflict_requires_review_unless_all_reject(self) -> None:
        group = {"representation_conflict": True}
        status, reasons = _group_status(
            group,
            [
                {"status": AUTO_ACCEPT, "reason_codes": []},
                {"status": AUTO_REJECT, "reason_codes": ["prefix_parse_invalid"]},
            ],
        )
        self.assertEqual(status, REVIEW_REQUIRED)
        self.assertIn("representation_conflict", reasons)
        status, _ = _group_status(
            group,
            [{"status": AUTO_REJECT, "reason_codes": ["prefix_parse_invalid"]}],
        )
        self.assertEqual(status, AUTO_REJECT)

    def _make_pipeline(self, root: Path):
        run = root / "run"
        run.mkdir()
        expression = self._expression()
        metadata = {
            "schema": "factor_gfn.trainer.v1",
            "run_id": "run",
            "config_fingerprint": "a" * 64,
            "reward_provider_fingerprint": "b" * 64,
            "config_manifest": {
                "schema": "factor_gfn.gfn_config.v1",
                "config": {"training": {"seed": 42}},
                "token_space_fingerprint": action_space_fingerprint(),
            },
            "reward_provider": {
                "context_fingerprint": "c" * 64,
                "reward_config": {"test": True},
            },
        }
        _write_json(run / "run_metadata.json", metadata)
        row = {
            "request_index": 1,
            "branch": "main",
            "phase": "test",
            "logical_step": 1,
            "formula": expression.to_formula(),
            "prefix_token_ids": list(expression.to_prefix()),
            "structural_hash": expression.structural_hash(),
            "node_count": expression.stats.node_count,
            "depth": expression.stats.depth,
            "valid": True,
            "reward": 0.1,
            "rejection_reason": None,
        }
        (run / "evaluations.jsonl").write_text(
            json.dumps(row) + "\n", encoding="utf-8"
        )
        spec = CandidateSourceSpec(
            source_id="source",
            source_type="discovery_run",
            source_role="historical_discovery",
            source_path=run,
            approval_note="approved test source",
        )
        source_set = materialize_source_set([spec], root / "snapshots")
        registry = import_candidate_source_set(source_set, root / "registries")
        return source_set, registry

    def test_end_to_end_audit_is_immutable_and_metric_reuse_is_forbidden(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_set, registry = self._make_pipeline(root)
            first = audit_expression_compatibility(
                registry, source_set, root / "audits"
            )
            second = audit_expression_compatibility(
                registry, source_set, root / "audits"
            )
            self.assertEqual(first, second)
            manifest = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(manifest["counts"][AUTO_ACCEPT], 1)
            self.assertEqual(manifest["counts"][AUTO_REJECT], 0)
            self.assertEqual(manifest["counts"][REVIEW_REQUIRED], 0)
            accepted = json.loads(
                (first.parent / "auto_accepted_candidate_registry.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual(accepted["historical_metric_reuse"], "forbidden")
            self.assertTrue(accepted["stage6_metric_recompute_required"])

    def test_incomplete_input_registry_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_set, registry = self._make_pipeline(root)
            manifest = json.loads(registry.read_text(encoding="utf-8"))
            manifest["registry_status"] = "incomplete"
            registry.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "unresolved schema rejection"):
                audit_expression_compatibility(
                    registry, source_set, root / "audits"
                )


if __name__ == "__main__":
    unittest.main()
