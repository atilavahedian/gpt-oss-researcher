import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class ResearchPipelineTests(unittest.TestCase):
    def test_default_model_policy_prefers_gpt_oss(self):
        from gpt_oss_researcher.model_policy import default_model_policy

        policy = default_model_policy()

        self.assertEqual(policy.primary_model, "openai/gpt-oss-20b")
        self.assertIn("openai/gpt-oss-120b", policy.verifier_models)
        self.assertEqual(policy.region_preference, "us")
        self.assertTrue(policy.prefers_open_weights)

    def test_runner_writes_reproducible_evidence_artifacts(self):
        from gpt_oss_researcher.pipeline import run_research_pipeline
        from gpt_oss_researcher.templates import default_experiment_config

        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            result = run_research_pipeline(
                config=default_experiment_config(),
                output_dir=output_dir,
                provider_name="fixture",
            )

            self.assertEqual(result.metrics["strategies"]["verifier_rerank"]["accuracy"], 1.0)
            self.assertTrue((output_dir / "run_manifest.json").exists())
            self.assertTrue((output_dir / "predictions.jsonl").exists())
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "research_report.md").exists())

            manifest = json.loads((output_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["model_policy"]["primary_model"], "openai/gpt-oss-20b")
            self.assertEqual(manifest["provider"], "fixture")
            self.assertIn("dataset_sha256", manifest)

    def test_verifier_reranking_beats_single_answer_on_fixture_tasks(self):
        from gpt_oss_researcher.pipeline import run_research_pipeline
        from gpt_oss_researcher.templates import default_experiment_config

        with TemporaryDirectory() as tmp:
            result = run_research_pipeline(
                config=default_experiment_config(),
                output_dir=Path(tmp),
                provider_name="fixture",
            )

            single = result.metrics["strategies"]["single"]
            rerank = result.metrics["strategies"]["verifier_rerank"]
            self.assertLess(single["accuracy"], rerank["accuracy"])
            self.assertGreaterEqual(rerank["accuracy"], 0.8)

    def test_adaptive_compute_uses_fewer_samples_than_fixed_reranking(self):
        from gpt_oss_researcher.pipeline import run_research_pipeline
        from gpt_oss_researcher.templates import default_experiment_config

        with TemporaryDirectory() as tmp:
            result = run_research_pipeline(
                config=default_experiment_config(),
                output_dir=Path(tmp),
                provider_name="fixture",
            )

            fixed = result.metrics["strategies"]["verifier_rerank"]["avg_samples_per_task"]
            adaptive = result.metrics["strategies"]["adaptive_verifier_rerank"]["avg_samples_per_task"]
            self.assertLess(adaptive, fixed)
            self.assertGreaterEqual(
                result.metrics["strategies"]["adaptive_verifier_rerank"]["accuracy"],
                result.metrics["strategies"]["single"]["accuracy"],
            )

    def test_report_contains_hypothesis_results_and_limitations(self):
        from gpt_oss_researcher.pipeline import run_research_pipeline
        from gpt_oss_researcher.templates import default_experiment_config

        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            run_research_pipeline(
                config=default_experiment_config(),
                output_dir=output_dir,
                provider_name="fixture",
            )
            report = (output_dir / "research_report.md").read_text()

        self.assertIn("## Hypothesis", report)
        self.assertIn("## Results", report)
        self.assertIn("## Limitations", report)
        self.assertIn("verifier-guided test-time compute", report)


if __name__ == "__main__":
    unittest.main()

