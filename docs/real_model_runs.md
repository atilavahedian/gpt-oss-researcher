# Real Model Runs

The default fixture run is intentionally local and deterministic. Use it for repository verification.

For real model runs, install optional dependencies:

```bash
python3 -m pip install -e '.[models]'
```

## OpenAI Provider

```bash
OPENAI_API_KEY=... \
OPENAI_RESEARCH_MODEL=<openai-api-model> \
gpt-oss-researcher all --provider openai --output-dir results/openai_run
```

Use this provider for hosted OpenAI API models. For the open-weight `gpt-oss` model IDs, use the Transformers provider or a hosted inference service that exposes those Hugging Face repos.

## Transformers Provider

```bash
HF_RESEARCH_MODEL=openai/gpt-oss-20b \
gpt-oss-researcher all --provider transformers --output-dir results/hf_run
```

Notes:

- `openai/gpt-oss-20b` and `openai/gpt-oss-120b` are large models.
- Local Transformers runs need suitable GPU memory or a quantized/runtime-specific setup.
- Do not compare fixture metrics against real model metrics as if they measure the same thing.
