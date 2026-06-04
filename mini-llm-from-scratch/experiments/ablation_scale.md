# Model Scale Ablation

## Goal

Compare a sufficiently trained 134M model with the final 212M model under the same tokenizer, dataset, position encoding, and context length.

## Configs

- `configs/scale_134m_rope_ctx1024.yaml`
- `configs/final_212m_rope_ctx1024.yaml`

## Commands

```bash
./scripts/prepare_fineweb_edu.sh
CONFIGS="configs/scale_134m_rope_ctx1024.yaml configs/final_212m_rope_ctx1024.yaml" \
  ./scripts/run_experiment_suite.sh
```

## Metrics

- validation loss
- validation perplexity
- training tokens per second
- peak GPU memory
- fixed-prompt generation samples
