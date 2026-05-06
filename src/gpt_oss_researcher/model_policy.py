from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ModelCandidate:
    model_id: str
    role: str
    organization: str
    country: str
    open_weights: bool
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "role": self.role,
            "organization": self.organization,
            "country": self.country,
            "open_weights": self.open_weights,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ModelPolicy:
    primary_model: str
    verifier_models: List[str]
    fallback_models: List[str]
    region_preference: str
    prefers_open_weights: bool
    candidates: List[ModelCandidate]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_model": self.primary_model,
            "verifier_models": list(self.verifier_models),
            "fallback_models": list(self.fallback_models),
            "region_preference": self.region_preference,
            "prefers_open_weights": self.prefers_open_weights,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def default_model_policy() -> ModelPolicy:
    """Return the repo's default model selection policy.

    The policy is a preference order, not a claim that local tests load these
    large models. The fixture provider keeps verification cheap and reproducible.
    """

    candidates = [
        ModelCandidate(
            model_id="openai/gpt-oss-20b",
            role="primary_generator",
            organization="OpenAI",
            country="US",
            open_weights=True,
            notes="Default American open-weight generator target for real runs.",
        ),
        ModelCandidate(
            model_id="openai/gpt-oss-120b",
            role="verifier_or_large_generator",
            organization="OpenAI",
            country="US",
            open_weights=True,
            notes="Preferred larger verifier target when compute or hosted inference is available.",
        ),
        ModelCandidate(
            model_id="google/gemma-3-4b-it",
            role="fallback_generator",
            organization="Google",
            country="US",
            open_weights=True,
            notes="American fallback for lower-memory local experiments.",
        ),
        ModelCandidate(
            model_id="meta-llama/Llama-3.2-3B-Instruct",
            role="fallback_generator",
            organization="Meta",
            country="US",
            open_weights=True,
            notes="American fallback for broad local tool compatibility.",
        ),
    ]
    return ModelPolicy(
        primary_model="openai/gpt-oss-20b",
        verifier_models=["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
        fallback_models=["google/gemma-3-4b-it", "meta-llama/Llama-3.2-3B-Instruct"],
        region_preference="us",
        prefers_open_weights=True,
        candidates=candidates,
    )

