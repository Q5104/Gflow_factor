import json
import tempfile
import unittest
from pathlib import Path

from factor_gfn.backtest.selection import import_candidate_runs
from factor_gfn.grammar import Expression


class CandidateRegistryTests(unittest.TestCase):
    def _write_run(
        self,
        root: Path,
        name: str,
        *,
        provider_fingerprint: str = "a" * 64,
        context_fingerprint: str = "b" * 64,
        corrupt_formula: bool = False,
    ) -> Path:
        run_dir = root / name
        run_dir.mkdir()
        expression = Expression.from_prefix([3])
        formula = "open" if corrupt_formula else expression.to_formula()
        assignment = {
            "provider_fingerprint": provider_fingerprint,
            "expression_hash": expression.structural_hash(),
            "formula": formula,
            "prefix_token_ids": list(expression.to_prefix()),
            "node_count": expression.stats.node_count,
            "depth": expression.stats.depth,
            "reward_result": {
                "expression_hash": expression.structural_hash(),
                "valid": True,
                "reward": 0.01,
            },
        }
        base = {
            "request_index": 1,
            "branch": "main",
            "phase": "train_step_1",
            "formula": formula,
            "structural_hash": expression.structural_hash(),
            "prefix_token_ids": list(expression.to_prefix()),
            "node_count": expression.stats.node_count,
            "depth": expression.stats.depth,
            "valid": True,
            "reward": 0.01,
            "rejection_reason": None,
            "metadata": assignment,
        }
        replay = dict(base)
        replay["branch"] = "determinism_replay"
        replay["phase"] = "step_5"
        (run_dir / "evaluations.jsonl").write_text(
            "\n".join(json.dumps(row) for row in (base, replay)) + "\n",
            encoding="utf-8",
        )
        metadata = {
            "schema": "factor_gfn.trainer.v1",
            "run_id": name,
            "reward_provider_fingerprint": provider_fingerprint,
            "reward_provider": {
                "context_fingerprint": context_fingerprint,
                "evaluation_config": {"horizon": 5},
                "reward_config": {"industry_neutralization": True},
            },
        }
        (run_dir / "run_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        (run_dir / "experiment_manifest.json").write_text(
            json.dumps({"run_id": name, "main_reward_requests": 1}),
            encoding="utf-8",
        )
        (run_dir / "best_candidate.json").write_text(
            json.dumps({"candidate": {"structural_hash": expression.structural_hash()}}),
            encoding="utf-8",
        )
        return run_dir

    def test_import_audits_all_rows_but_registers_main_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._write_run(Path(temporary), "run_a")
            registry = import_candidate_runs([run_dir])
        self.assertEqual(len(registry.candidates), 1)
        self.assertIsNone(registry.candidates[0].train_long_direction)
        self.assertEqual(len(registry.candidates[0].origins), 1)
        self.assertFalse(
            registry.candidates[0].origins[0].has_neutralization_skip_diagnostics
        )
        audit = registry.run_audits[0]
        self.assertEqual(audit.total_evaluation_rows, 2)
        self.assertEqual(audit.candidate_rows, 1)
        self.assertEqual(audit.ignored_rows, 1)
        self.assertIn("experiment_manifest.json", audit.optional_files_present)
        self.assertEqual(len(registry.fingerprint), 64)

    def test_formula_token_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._write_run(
                Path(temporary), "run_bad", corrupt_formula=True
            )
            with self.assertRaisesRegex(ValueError, "公式与 prefix Token"):
                import_candidate_runs([run_dir])

    def test_incompatible_sources_require_explicit_grouping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._write_run(root, "run_a")
            second = self._write_run(
                root,
                "run_b",
                provider_fingerprint="c" * 64,
                context_fingerprint="d" * 64,
            )
            with self.assertRaisesRegex(ValueError, "不兼容"):
                import_candidate_runs([first, second])
            registry = import_candidate_runs(
                [first, second], allow_mixed_contexts=True
            )
        self.assertEqual(len(registry.compatibility_groups), 2)
        self.assertEqual(len(registry.candidates), 1)
        self.assertEqual(len(registry.candidates[0].origins), 2)


if __name__ == "__main__":
    unittest.main()
