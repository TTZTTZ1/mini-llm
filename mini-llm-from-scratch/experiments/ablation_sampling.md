# Sampling Strategy Ablation

## Goal

Compare greedy, temperature, top-k, and top-p decoding on the same final checkpoint.

## Command

```bash
./scripts/run_posttrain_evals.sh
```

## Metrics

- repetition
- topic consistency
- diversity
- manually inspected fixed-prompt samples
