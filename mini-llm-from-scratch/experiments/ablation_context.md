# Context Length Ablation

## Goal

Compare context lengths 512, 1024, and 1536 for the same 212M RoPE model. Batch sizes are chosen so each step sees 24576 tokens:

- ctx 512: batch 48
- ctx 1024: batch 24
- ctx 1536: batch 16

## Configs

- `configs/ablation_212m_context_512.yaml`
- `configs/final_212m_rope_ctx1024.yaml`
- `configs/ablation_212m_context_1536.yaml`

## Commands

```bash
./scripts/prepare_fineweb_edu.sh
CONFIGS="configs/ablation_212m_context_512.yaml configs/final_212m_rope_ctx1024.yaml configs/ablation_212m_context_1536.yaml" \
  ./scripts/run_experiment_suite.sh
```

Before the full ctx1536 run, do a short safety run:

```bash
CONFIGS="configs/ablation_212m_context_1536.yaml" MAX_STEPS_OVERRIDE=20 ./scripts/run_experiment_suite.sh
```

## Metrics

- validation loss and perplexity
- tokens per second
- peak GPU memory
- long-prompt generation behavior
