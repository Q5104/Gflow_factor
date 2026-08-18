import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from factor_gfn.backtest.sources import (
    CandidateSourceSpec,
    materialize_candidate_source,
    materialize_source_set,
)
from factor_gfn.backtest import sources as source_module


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _row(index: int, *, branch: str = "main", source: str | None = None):
    row = {
        "request_index": index,
        "branch": branch,
        "formula": "open",
        "prefix_token_ids": [3],
        "structural_hash": f"{index:064x}",
        "node_count": 1,
        "depth": 0,
        "valid": True,
    }
    if source is not None:
        row["source"] = source
    return row


class CandidateSourceSnapshotTests(unittest.TestCase):
    def _make_run(
        self,
        root: Path,
        name: str,
        *,
        records: int = 3,
        committed: int | None = None,
        active: bool = False,
        with_state: bool = True,
    ) -> Path:
        run = root / name
        run.mkdir()
        metadata = {
            "schema": "factor_gfn.trainer.v1",
            "run_id": name,
            "config_fingerprint": "a" * 64,
            "reward_provider_fingerprint": "b" * 64,
            "config_manifest": {"config": {"training": {"seed": 42}}},
            "reward_provider": {
                "context_fingerprint": "c" * 64,
                "reward_config": {"industry_neutralization": True},
            },
        }
        _write_json(run / "run_metadata.json", metadata)
        (run / "evaluations.jsonl").write_text(
            "".join(json.dumps(_row(index)) + "\n" for index in range(1, records + 1)),
            encoding="utf-8",
        )
        if with_state:
            count = records if committed is None else committed
            _write_json(
                run / "run_state.json",
                {
                    "run_id": name,
                    "status": "running" if active else "ready",
                    "active_step": 2 if active else None,
                    "current_step": 1,
                    "optimizer_step": 1,
                    "evaluation_records": count,
                    "step_metric_records": 1,
                    "updated_at_utc": "2026-08-14T00:00:00+00:00",
                },
            )
        return run

    def _spec(self, run: Path, source_id: str = "run_a") -> CandidateSourceSpec:
        return CandidateSourceSpec(
            source_id=source_id,
            source_type="discovery_run",
            source_role="historical_discovery",
            source_path=run,
            approval_note="approved historical discovery source",
        )

    def test_active_run_uses_committed_record_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self._make_run(root, "run_a", records=3, committed=2, active=True)
            manifest_path = materialize_candidate_source(
                self._spec(run), root / "snapshots"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            captured = manifest_path.parent / "evaluations.jsonl"
            self.assertEqual(len(captured.read_text(encoding="utf-8").splitlines()), 2)
            self.assertEqual(manifest["snapshot_kind"], "committed_jsonl_prefix")
            self.assertEqual(manifest["cutoff"]["committed_evaluation_records"], 2)
            self.assertEqual(manifest["source_observation"]["size_after"], (run / "evaluations.jsonl").stat().st_size)

    def test_active_run_may_advance_during_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self._make_run(root, "run_a", records=3, committed=2, active=True)
            original = source_module._read_jsonl_prefix

            def read_then_advance(path: Path, count: int) -> bytes:
                result = original(path, count)
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(_row(4)) + "\n")
                state = json.loads((run / "run_state.json").read_text(encoding="utf-8"))
                state.update(
                    current_step=2,
                    optimizer_step=2,
                    evaluation_records=4,
                    step_metric_records=2,
                    active_step=3,
                    updated_at_utc="2026-08-14T00:01:00+00:00",
                )
                _write_json(run / "run_state.json", state)
                return result

            with patch.object(source_module, "_read_jsonl_prefix", read_then_advance):
                manifest_path = materialize_candidate_source(
                    self._spec(run), root / "snapshots"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["cutoff"]["committed_evaluation_records"], 2)
            self.assertGreater(
                manifest["source_observation"]["size_after"],
                manifest["source_observation"]["size_before"],
            )

    def test_committed_count_beyond_complete_lines_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self._make_run(root, "run_a", records=2, committed=3, active=True)
            with self.assertRaisesRegex(RuntimeError, "声明 3 条评价"):
                materialize_candidate_source(self._spec(run), root / "snapshots")

    def test_stable_run_requires_state_count_to_match_full_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self._make_run(root, "run_a", records=3, committed=2, active=False)
            with self.assertRaisesRegex(RuntimeError, "与 JSONL 行数不一致"):
                materialize_candidate_source(self._spec(run), root / "snapshots")

    def test_diagnostic_audit_materializes_only_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            diagnostic = root / "manual_diagnostic_seed42"
            diagnostic.mkdir()
            rows = [
                _row(1, source="exhaustive_full_evaluation"),
                _row(2, source="calibration"),
                _row(3, source="discovery"),
                _row(4, source="discovery"),
            ]
            rows[3]["legacy_nonfinite_metric"] = float("nan")
            (diagnostic / "candidate_audit.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
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
                source_id="diagnostic_discovery",
                source_type="diagnostic_audit",
                source_role="historical_discovery",
                source_path=diagnostic,
                approval_note="approved 6/20 diagnostic discovery source",
                included_record_sources=("discovery",),
                excluded_record_sources=("calibration", "exhaustive_full_evaluation"),
            )
            manifest_path = materialize_candidate_source(spec, root / "snapshots")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["cutoff"]["raw_records"], 4)
            self.assertEqual(manifest["cutoff"]["materialized_records"], 2)
            captured = (manifest_path.parent / "candidate_audit.jsonl").read_text(encoding="utf-8")
            self.assertEqual(len(captured.splitlines()), 2)
            self.assertTrue(all(json.loads(line)["source"] == "discovery" for line in captured.splitlines()))

    def _make_registry(self, path: Path) -> None:
        connection = sqlite3.connect(path)
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
                    structural_hash TEXT PRIMARY KEY,
                    node_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    provider_fingerprint TEXT NOT NULL,
                    context_fingerprint TEXT NOT NULL
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
                rows.append(
                    (
                        f"{index:064x}",
                        1 if index < 6 else 2,
                        "evaluated",
                        "b" * 64,
                        "c" * 64,
                    )
                )
            connection.executemany("INSERT INTO candidates VALUES (?, ?, ?, ?, ?)", rows)
            connection.commit()
        finally:
            connection.close()

    def test_exhaustive_registry_uses_consistent_backup_and_logical_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry_dir = root / "manual_diagnostic_seed42"
            registry_dir.mkdir()
            self._make_registry(registry_dir / "exhaustive_registry.sqlite3")
            spec = CandidateSourceSpec(
                source_id="exact_n1_n2",
                source_type="exhaustive_registry",
                source_role="exhaustive",
                source_path=registry_dir,
                approval_note="approved exact N=1/2 registry",
            )
            first = materialize_candidate_source(spec, root / "snapshots")
            second = materialize_candidate_source(spec, root / "snapshots")
            self.assertEqual(first, second)
            manifest = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(manifest["record_counts"]["records"], 642)
            self.assertEqual(len(manifest["logical_content_fingerprint"]), 64)
            self.assertTrue((first.parent / "exhaustive_registry.sqlite3").is_file())

    def test_source_set_fingerprint_is_order_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_a = self._make_run(root, "run_a", records=1, with_state=False)
            run_b = self._make_run(root, "run_b", records=1, with_state=False)
            first_spec = self._spec(run_a, "source_a")
            second_spec = self._spec(run_b, "source_b")
            first = materialize_source_set(
                [first_spec, second_spec], root / "snapshots"
            )
            second = materialize_source_set(
                [second_spec, first_spec], root / "snapshots"
            )
            self.assertEqual(first, second)

    def test_unapproved_source_cannot_enter_source_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self._make_run(root, "run_a", records=1, with_state=False)
            spec = CandidateSourceSpec(
                source_id="pending",
                source_type="discovery_run",
                source_role="historical_discovery",
                source_path=run,
                inclusion_status="pending_review",
                approval_note="awaiting approval",
            )
            with self.assertRaisesRegex(ValueError, "未批准来源"):
                materialize_source_set([spec], root / "snapshots")


if __name__ == "__main__":
    unittest.main()
