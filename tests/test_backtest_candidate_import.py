import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from factor_gfn.backtest.candidate_import import import_candidate_source_set
from factor_gfn.backtest.sources import CandidateSourceSpec, materialize_source_set


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _row(
    structural_hash: str,
    *,
    formula: str = "open",
    prefix=None,
    node_count: int = 1,
    depth: int = 0,
):
    return {
        "request_index": 1,
        "branch": "main",
        "formula": formula,
        "prefix_token_ids": [0] if prefix is None else prefix,
        "structural_hash": structural_hash,
        "node_count": node_count,
        "depth": depth,
        "valid": True,
        "reward": 0.1,
        "rejection_reason": None,
    }


def _formal_manifest() -> dict:
    return {
        "schema": "factor_gfn.gfn_config.no_anchor.v1",
        "config": {"training": {"seed": 42}},
        "strata": {
            "schema": "factor_gfn.complexity_conditioned_no_anchor.v1",
            "normal_discovery_equals_feasible": True,
        },
        "state_adapter": {
            "schema": "factor_gfn.state_adapter.v2",
            "condition_features": ["target_node_count/max_nodes"],
        },
    }


class CandidateImportTests(unittest.TestCase):
    def _make_run(
        self,
        root: Path,
        name: str,
        rows,
        *,
        formal: bool = False,
    ) -> Path:
        run = root / name
        run.mkdir()
        metadata = {
            "schema": "factor_gfn.trainer.v1",
            "run_id": name,
            "config_fingerprint": "a" * 64,
            "reward_provider_fingerprint": "b" * 64,
            "config_manifest": (
                _formal_manifest()
                if formal
                else {
                    "schema": "factor_gfn.gfn_config.v1",
                    "config": {"training": {"seed": 42}},
                }
            ),
            "reward_provider": {
                "context_fingerprint": "c" * 64,
                "reward_config": {"test": True},
            },
        }
        _write_json(run / "run_metadata.json", metadata)
        (run / "evaluations.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return run

    def _spec(self, run: Path, source_id: str, *, formal: bool = False):
        return CandidateSourceSpec(
            source_id=source_id,
            source_type="discovery_run",
            source_role="formal_discovery" if formal else "historical_discovery",
            source_path=run,
            approval_note="approved test discovery source",
        )

    def _import(self, root: Path, specs):
        source_set = materialize_source_set(specs, root / "snapshots")
        manifest = import_candidate_source_set(source_set, root / "registries")
        return manifest, json.loads(manifest.read_text(encoding="utf-8"))

    def test_formal_target_is_derived_but_historical_target_is_null(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            structural_hash = "1" * 64
            formal = self._make_run(
                root, "formal", [_row(structural_hash, node_count=3)], formal=True
            )
            historical = self._make_run(
                root, "historical", [_row(structural_hash, node_count=3)]
            )
            manifest_path, manifest = self._import(
                root,
                [
                    self._spec(formal, "formal", formal=True),
                    self._spec(historical, "historical"),
                ],
            )
            origins = [
                json.loads(line)
                for line in (manifest_path.parent / "normalized_candidate_origins.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            by_source = {row["provenance"]["source_id"]: row for row in origins}
            self.assertEqual(by_source["formal"]["target_node_count"], 3)
            self.assertEqual(
                by_source["formal"]["target_node_count_method"],
                "derived_from_exact_n_no_anchor_formal_contract",
            )
            self.assertIsNone(by_source["historical"]["target_node_count"])
            self.assertEqual(manifest["counts"]["normalized_origins"], 2)
            self.assertEqual(manifest["counts"]["claimed_hash_groups"], 1)
            self.assertEqual(manifest["counts"]["duplicate_origins"], 1)

    def test_formal_source_without_exact_n_contract_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self._make_run(root, "formal", [_row("2" * 64)])
            source_set = materialize_source_set(
                [self._spec(run, "formal", formal=True)], root / "snapshots"
            )
            with self.assertRaisesRegex(RuntimeError, "exact-N no-anchor contract"):
                import_candidate_source_set(source_set, root / "registries")

    def test_schema_rejection_completes_audit_but_blocks_downstream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = _row("3" * 64)
            invalid = _row("4" * 64)
            invalid.pop("formula")
            run = self._make_run(root, "history", [valid, invalid])
            _, manifest = self._import(root, [self._spec(run, "history")])
            self.assertEqual(manifest["registry_status"], "incomplete")
            self.assertFalse(manifest["downstream_eligible"])
            self.assertEqual(manifest["counts"]["schema_rejected"], 1)
            self.assertIn(
                "unresolved_schema_rejections", manifest["downstream_block_reasons"]
            )
            self.assertEqual(
                manifest["digests"]["schema_rejection_ledger"],
                manifest["fingerprint_payload"]["schema_rejection_ledger_digest"],
            )

    def test_representation_conflict_is_preserved_and_blocks_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            structural_hash = "5" * 64
            first = self._make_run(root, "first", [_row(structural_hash)])
            second = self._make_run(
                root,
                "second",
                [_row(structural_hash, formula="close", prefix=[3])],
            )
            manifest_path, manifest = self._import(
                root, [self._spec(first, "first"), self._spec(second, "second")]
            )
            group = json.loads(
                (manifest_path.parent / "candidate_registry.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertTrue(group["representation_conflict"])
            self.assertFalse(group["downstream_eligible"])
            self.assertEqual(len(group["representations"]), 2)
            self.assertEqual(manifest["counts"]["representation_conflicts"], 1)
            self.assertFalse(manifest["downstream_eligible"])

    def test_snapshot_artifact_mutation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self._make_run(root, "history", [_row("6" * 64)])
            source_set = materialize_source_set(
                [self._spec(run, "history")], root / "snapshots"
            )
            source_manifest = next(
                (root / "snapshots" / "sources" / "history").glob(
                    "*/source_snapshot.json"
                )
            )
            with (source_manifest.parent / "evaluations.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(_row("7" * 64)) + "\n")
            with self.assertRaisesRegex(RuntimeError, "artifact 指纹不符"):
                import_candidate_source_set(source_set, root / "registries")

    def test_source_set_fingerprint_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self._make_run(root, "history", [_row("9" * 64)])
            source_set = materialize_source_set(
                [self._spec(run, "history")], root / "snapshots"
            )
            manifest = json.loads(source_set.read_text(encoding="utf-8"))
            manifest["mode"] = "final"
            source_set.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "source-set fingerprint 不符"):
                import_candidate_source_set(source_set, root / "registries")

    def test_diagnostic_adapter_preserves_compact_old_metric_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            diagnostic = root / "diagnostic"
            diagnostic.mkdir()
            row = _row("8" * 64)
            row["source"] = "discovery"
            (diagnostic / "candidate_audit.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )
            _write_json(
                diagnostic / "diagnostic_context.json",
                {
                    "schema": "factor_gfn.conditional_diagnostic.v1",
                    "config_fingerprint": "a" * 64,
                    "provider_fingerprint": "b" * 64,
                    "context_fingerprint": "c" * 64,
                },
            )
            spec = CandidateSourceSpec(
                source_id="diagnostic",
                source_type="diagnostic_audit",
                source_role="historical_discovery",
                source_path=diagnostic,
                approval_note="approved diagnostic discovery source",
                included_branches=(),
                included_record_sources=("discovery",),
            )
            manifest_path, manifest = self._import(root, [spec])
            origin = json.loads(
                (manifest_path.parent / "normalized_candidate_origins.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual(origin["old_metric_audit"]["old_reward"], 0.1)
            self.assertFalse(
                origin["old_metric_audit"]["reuse_for_stage6_selection"]
            )
            self.assertEqual(manifest["counts"]["normalized_origins"], 1)

    def test_exhaustive_adapter_imports_frozen_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry_dir = root / "registry"
            registry_dir.mkdir()
            database = registry_dir / "exhaustive_registry.sqlite3"
            connection = sqlite3.connect(database)
            try:
                connection.executescript(
                    """
                    CREATE TABLE metadata (key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
                    CREATE TABLE strata (
                        node_count INTEGER PRIMARY KEY,
                        expected_canonical_count INTEGER NOT NULL,
                        enumeration_complete INTEGER NOT NULL,
                        exact_status TEXT NOT NULL
                    );
                    CREATE TABLE candidates (
                        structural_hash TEXT PRIMARY KEY, source TEXT NOT NULL,
                        node_count INTEGER NOT NULL, depth INTEGER NOT NULL,
                        formula TEXT NOT NULL, prefix_token_ids_json TEXT NOT NULL,
                        provider_fingerprint TEXT NOT NULL,
                        context_fingerprint TEXT NOT NULL, status TEXT NOT NULL,
                        valid INTEGER, rejection_reason TEXT,
                        reward_details_json TEXT, target_mass REAL
                    );
                    """
                )
                connection.executemany(
                    "INSERT INTO metadata VALUES (?, ?)",
                    [
                        ("schema", json.dumps("factor_gfn.exhaustive_registry.v2")),
                        ("plan_fingerprint", json.dumps("d" * 64)),
                    ],
                )
                connection.executemany(
                    "INSERT INTO strata VALUES (?, ?, ?, ?)",
                    [(1, 6, 1, "complete"), (2, 636, 1, "complete")],
                )
                rows = []
                for index in range(642):
                    node_count = 1 if index < 6 else 2
                    rows.append(
                        (
                            f"{index:064x}",
                            "exhaustive_full_evaluation",
                            node_count,
                            node_count - 1,
                            f"factor_{index}",
                            json.dumps([index]),
                            "b" * 64,
                            "c" * 64,
                            "evaluated",
                            1,
                            None,
                            json.dumps({"reward_result": {"reward": 0.2}}),
                            0.2,
                        )
                    )
                connection.executemany(
                    "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
                connection.commit()
            finally:
                connection.close()
            spec = CandidateSourceSpec(
                source_id="exhaustive",
                source_type="exhaustive_registry",
                source_role="exhaustive",
                source_path=registry_dir,
                approval_note="approved exhaustive source",
                included_record_sources=("exhaustive_full_evaluation",),
            )
            manifest_path, manifest = self._import(root, [spec])
            first = json.loads(
                (manifest_path.parent / "normalized_candidate_origins.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(manifest["counts"]["normalized_origins"], 642)
            self.assertIsNone(first["target_node_count"])
            self.assertEqual(
                first["target_node_count_method"],
                "not_applicable_exhaustive_enumeration",
            )


if __name__ == "__main__":
    unittest.main()
