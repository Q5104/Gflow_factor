import json
import unittest
from pathlib import Path


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "archive"
    / "diagnostics"
    / "run_complexity_conditioned_diagnostic_6_20.ipynb"
)


class HistoricalComplexityDiagnosticNotebookTests(unittest.TestCase):
    def test_notebook_is_explicitly_historical_and_not_an_active_resume_entrypoint(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        first_markdown = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        self.assertIn("历史诊断归档", first_markdown)
        self.assertIn("禁止恢复", first_markdown)
        self.assertIn("anchor", first_markdown.lower())


if __name__ == "__main__":
    unittest.main()
