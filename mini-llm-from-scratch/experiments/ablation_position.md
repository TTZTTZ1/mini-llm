# Position Encoding Ablation

## Goal

Compare RoPE, learned absolute position embeddings, and no position encoding on the same 212M architecture and 3.2B-token budget.

## Configs

- `configs/final_212m_rope_ctx1024.yaml`
- `configs/ablation_212m_pos_abs.yaml`
- `configs/ablation_212m_pos_none.yaml`

## Commands

```bash
./scripts/prepare_fineweb_edu.sh
CONFIGS="configs/final_212m_rope_ctx1024.yaml configs/ablation_212m_pos_abs.yaml configs/ablation_212m_pos_none.yaml" \
  ./scripts/run_experiment_suite.sh
```

## Metrics

- validation loss and perplexity at matched token counts
- training stability
- generation quality on fixed prompts
