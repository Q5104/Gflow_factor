import unittest
from pathlib import Path

from factor_gfn.gfn.no_anchor_search_runner import (
    NO_ANCHOR_REAL_SEARCH_SCHEMA,
    _require_frozen_settings,
)
from factor_gfn.gfn.search_runner import RealSearchSettings


class NoAnchorSearchRunnerContractTests(unittest.TestCase):
    def test_schema_is_distinct_from_legacy_real_search(self):
        self.assertEqual(
            NO_ANCHOR_REAL_SEARCH_SCHEMA,
            "factor_gfn.no_anchor_real_search.v1",
        )

    def test_formal_settings_freeze_1000_steps_and_seed42(self):
        valid = RealSearchSettings(
            max_steps=1000,
            seed=42,
            device="cuda:0",
            run_root=Path("runs/stage5_no_anchor_formal_6_20"),
        )
        _require_frozen_settings(valid)
        with self.assertRaisesRegex(ValueError, "max_steps"):
            _require_frozen_settings(
                RealSearchSettings(max_steps=10, seed=42, device="cuda:0")
            )
        with self.assertRaisesRegex(ValueError, "seed"):
            _require_frozen_settings(
                RealSearchSettings(max_steps=1000, seed=7, device="cuda:0")
            )


if __name__ == "__main__":
    unittest.main()
