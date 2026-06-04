# KV Cache Benchmark

## Goal

Measure autoregressive generation speed with and without KV Cache using the final 212M checkpoint.

## Command

```bash
./scripts/run_posttrain_evals.sh
```

## Metrics

- tokens per second
- peak GPU memory
- speedup ratio between cached and uncached generation
