# Quick Start Guide

Get started with the Korean ASR Benchmark in 3 steps.

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Run Benchmark

```bash
python -m src.main
```

This will:

- Download FLEURS Korean dataset (~2GB, cached locally)
- Test 17 ASR models
- Generate rankings and reports in `results/`

## 3. View Results

Check the results:

```bash
cat results/top_5_models.txt
```

Full report:

```bash
cat results/benchmark_report.md
```

## Command Reference

```bash
# List all models
python -m src.main --list-models

# Test single model (faster)
python -m src.main --model whisper-large-v3

# Get top 10 instead of top 5
python -m src.main --top-n 10

# Run tests
python -m unittest discover tests/
```

## Expected Runtime

- Single model: 2-10 minutes (depending on model size and hardware)
- All 17 models: 1-3 hours on GPU, 3-6 hours on CPU

## Minimal Config for Testing

To test quickly, edit `config.yaml`:

```yaml
benchmark:
  test_samples: 20 # Use only 20 samples instead of 100
  device: 'cpu' # or "cpu"
```

## Key Files

| File                          | Description                          |
| ----------------------------- | ------------------------------------ |
| `config.yaml`                 | Configuration for models and dataset |
| `results/top_5_models.txt`    | Top 5 ranked models                  |
| `results/benchmark_report.md` | Comprehensive report                 |
| `results/results.csv`         | All metrics in CSV format            |

## Next Steps

See [README.md](README.md) for detailed documentation.
