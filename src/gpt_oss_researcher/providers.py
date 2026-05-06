import os
from typing import List, Protocol

from gpt_oss_researcher.model_policy import ModelPolicy
from gpt_oss_researcher.schemas import Candidate, Task


class ModelProvider(Protocol):
    name: str

    def generate(self, task: Task, sample_count: int) -> List[Candidate]:
        ...


class FixtureProvider:
    name = "fixture"

    def __init__(self, model_policy: ModelPolicy):
        self.model_policy = model_policy

    def generate(self, task: Task, sample_count: int) -> List[Candidate]:
        if not task.fixture_candidates:
            pool = [task.answer]
        else:
            pool = list(task.fixture_candidates)
        candidates: List[Candidate] = []
        for index in range(sample_count):
            text = pool[index % len(pool)]
            candidates.append(Candidate(text=text, sample_index=index, model=self.model_policy.primary_model))
        return candidates


class OpenAIResponsesProvider:
    name = "openai"

    def __init__(self, model_policy: ModelPolicy):
        self.model_policy = model_policy
        self.model = os.environ.get("OPENAI_RESEARCH_MODEL")

    def generate(self, task: Task, sample_count: int) -> List[Candidate]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the models extra to use the OpenAI provider: pip install -e '.[models]'") from exc
        if "OPENAI_API_KEY" not in os.environ:
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAI provider")
        if not self.model:
            raise RuntimeError("OPENAI_RESEARCH_MODEL must name an OpenAI API model for the OpenAI provider")

        client = OpenAI()
        prompt = (
            "Answer the benchmark question concisely. Return only the final answer.\n\n"
            f"Question: {task.question}"
        )
        candidates: List[Candidate] = []
        for index in range(sample_count):
            response = client.responses.create(model=self.model, input=prompt, temperature=0.7)
            candidates.append(Candidate(text=response.output_text.strip(), sample_index=index, model=self.model))
        return candidates


class TransformersProvider:
    name = "transformers"

    def __init__(self, model_policy: ModelPolicy):
        self.model_policy = model_policy
        self.model_id = os.environ.get("HF_RESEARCH_MODEL", model_policy.primary_model)
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the models extra to use the Transformers provider: pip install -e '.[models]'") from exc

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="auto")

    def generate(self, task: Task, sample_count: int) -> List[Candidate]:
        self._load()
        assert self._tokenizer is not None
        assert self._model is not None
        prompt = f"Answer the benchmark question concisely.\nQuestion: {task.question}\nAnswer:"
        inputs = self._tokenizer(prompt, return_tensors="pt")
        candidates: List[Candidate] = []
        for index in range(sample_count):
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = text.split("Answer:", 1)[-1].strip()
            candidates.append(Candidate(text=answer, sample_index=index, model=self.model_id))
        return candidates


def build_provider(name: str, model_policy: ModelPolicy) -> ModelProvider:
    if name == "fixture":
        return FixtureProvider(model_policy)
    if name == "openai":
        return OpenAIResponsesProvider(model_policy)
    if name == "transformers":
        return TransformersProvider(model_policy)
    raise ValueError(f"Unknown provider: {name}")
