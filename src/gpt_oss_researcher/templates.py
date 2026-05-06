from gpt_oss_researcher.schemas import ExperimentConfig, Task


def default_experiment_config() -> ExperimentConfig:
    tasks = [
        Task(
            id="arithmetic_13",
            question="What is 8 + 5?",
            answer="13",
            rubric_keywords=["13"],
            fixture_candidates=["12", "13", "14", "13"],
        ),
        Task(
            id="capital_california",
            question="What is the capital of California?",
            answer="Sacramento",
            rubric_keywords=["Sacramento"],
            fixture_candidates=["Los Angeles", "Sacramento", "San Francisco", "Sacramento"],
        ),
        Task(
            id="space_needle",
            question="Which U.S. city is the Space Needle in?",
            answer="Seattle",
            rubric_keywords=["Seattle"],
            fixture_candidates=["Seattle", "Spokane", "Seattle", "Portland"],
        ),
        Task(
            id="prime_boolean",
            question="Is 29 a prime number? Answer true or false.",
            answer="true",
            rubric_keywords=["true"],
            fixture_candidates=["false", "true", "true", "false"],
        ),
        Task(
            id="declaration_year",
            question="In what year was the U.S. Declaration of Independence adopted?",
            answer="1776",
            rubric_keywords=["1776"],
            fixture_candidates=["1789", "1776", "1776", "1775"],
        ),
    ]
    return ExperimentConfig(
        title="Verifier-Guided Test-Time Compute on Bounded QA",
        research_question=(
            "Can verifier-guided test-time compute improve answer accuracy over a single "
            "generation while making the extra sample cost explicit?"
        ),
        hypothesis=(
            "If the correct answer appears in a sampled candidate set, verifier reranking "
            "will outperform a single-answer baseline; adaptive reranking will retain most "
            "of the gain while using fewer samples than fixed reranking."
        ),
        tasks=tasks,
        strategies=["single", "self_consistency", "verifier_rerank", "adaptive_verifier_rerank"],
        sample_count=4,
        adaptive_threshold=0.99,
    )

