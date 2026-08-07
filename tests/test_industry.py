import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from factor_gfn.data.industry import load_sw_level1_industries


class IndustryAlignmentTests(unittest.TestCase):
    def test_level1_labels_align_to_stock_list_and_keep_missing(self) -> None:
        frame = pd.DataFrame(
            {
                "stock_code": ["300033", "300033", "000001"],
                "industry_name": ["计算机", "软件开发", "银行"],
                "industry_type": ["申万一级", "申万二级", "申万一级"],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "industry_sw.parquet"
            frame.to_parquet(path, index=False)
            labels = load_sw_level1_industries(
                np.array(["000001", "000002", "300033"]),
                path,
            )

        self.assertEqual(labels.tolist(), ["银行", None, "计算机"])


if __name__ == "__main__":
    unittest.main()
