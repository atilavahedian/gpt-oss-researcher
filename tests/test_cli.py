import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_cli_all_generates_research_artifacts(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "artifacts"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT / "src")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "gpt_oss_researcher.cli",
                    "all",
                    "--output-dir",
                    str(output_dir),
                    "--provider",
                    "fixture",
                ],
                cwd=str(ROOT),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("research_report.md", completed.stdout)
            self.assertTrue((output_dir / "research_report.md").exists())
            metrics = json.loads((output_dir / "metrics.json").read_text())
            self.assertIn("verifier_rerank", metrics["strategies"])


if __name__ == "__main__":
    unittest.main()
