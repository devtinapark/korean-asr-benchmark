# Architecture Overview

This document describes the architecture of the Korean ASR Benchmark system.

## Repository Structure

```
korean-asr-benchmark/
├── config.yaml                 # Main configuration file (17 models configured)
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation
├── QUICKSTART.md              # Quick start guide
├── ARCHITECTURE.md            # This file
│
├── src/                        # Main source code
│   ├── __init__.py            # Package initialization
│   ├── main.py                # CLI entry point
│   ├── benchmark.py           # Benchmark orchestrator
│   ├── data_loader.py         # FLEURS dataset loader
│   ├── model_wrapper.py       # Unified ASR model interface
│   ├── metrics.py             # CER, WER, loanword metrics
│   ├── loanword_detector.py   # Konglish detection
│   ├── ranking.py             # Model ranking system
│   └── reporter.py            # Results reporting
│
├── tests/                      # Unit tests
│   ├── test_metrics.py
│   ├── test_loanword_detector.py
│   └── test_ranking.py
│
└── results/                    # Generated outputs (created on run)
    ├── benchmark_report.md
    ├── results.csv
    ├── results.json
    ├── results.yaml
    ├── top_5_models.txt
    └── predictions/
```

## System Components

### 1. Data Loading (`data_loader.py`)

**Classes:**

- `FleursKoreanLoader`: Loads FLEURS Korean dataset from HuggingFace
- `AudioPreprocessor`: Resamples and normalizes audio
- `AudioSample`: Data container for audio + transcription

**Key Features:**

- Automatic dataset download and caching
- Audio preprocessing (resampling, normalization)
- Configurable sample size for testing

### 2. Model Wrapper (`model_wrapper.py`)

**Classes:**

- `ASRModelWrapper`: Unified interface for different ASR architectures

**Supported Model Types:**

- Whisper (OpenAI)
- Wav2Vec2 (Facebook/Meta)
- MMS (Meta Multilingual)
- Hubert (Facebook)
- Custom fine-tuned models

**Key Features:**

- Automatic model type detection
- Fallback loading mechanisms
- CPU/GPU support
- Batch transcription

### 3. Metrics Calculation (`metrics.py`)

**Classes:**

- `KoreanMetricsCalculator`: CER, WER, precision, recall
- `LoanwordAccuracyCalculator`: Loanword-specific metrics
- `InferenceSpeedCalculator`: Speed benchmarking

**Computed Metrics:**

- Character Error Rate (CER)
- Word Error Rate (WER)
- CER/WER Ratio (Korean language validation)
- Loanword accuracy
- Character/Word precision & recall
- Samples per second
- Real-time factor (optional)

### 4. Loanword Detection (`loanword_detector.py`)

**Classes:**

- `KonglishDetector`: Identifies English loanwords in Korean text
- `LoanwordSubsetAnalyzer`: Analyzes performance on loanword-containing samples

**Features:**

- Kitchen-specific loanword dictionary
- Romanization filtering
- Distribution analysis
- Subset creation for focused evaluation

### 5. Benchmark Runner (`benchmark.py`)

**Classes:**

- `ASRBenchmark`: Main orchestrator
- `BenchmarkResult`: Result container

**Workflow:**

1. Load configuration
2. Initialize dataset
3. For each model:
   - Load model
   - Run inference on all samples
   - Calculate metrics
   - Store results
4. Return aggregated results

### 6. Ranking System (`ranking.py`)

**Classes:**

- `ModelRanker`: Ranks models based on composite score
- `RankingCriteria`: Configurable weights

**Ranking Formula:**

```
Composite Score =
  0.40 × CER_normalized +
  0.20 × WER_normalized +
  0.20 × Loanword_normalized +
  0.10 × Speed_normalized +
  0.10 × Ratio_normalized
```

**Process:**

1. Normalize all metrics to 0-1 scale
2. Apply weights
3. Calculate composite score
4. Sort by score (descending)
5. Select top N

### 7. Results Reporter (`reporter.py`)

**Classes:**

- `BenchmarkReporter`: Generates reports in multiple formats
- `PredictionExporter`: Exports detailed predictions

**Output Formats:**

- Markdown (comprehensive report)
- CSV (tabular data)
- JSON (structured data)
- YAML (structured data)
- TXT (top N list)

## Data Flow

```
┌─────────────┐
│ config.yaml │
└──────┬──────┘
       │
       v
┌─────────────────┐     ┌──────────────┐
│ ASRBenchmark    │────>│ FLEURS Data  │
│ (main.py)       │     │ Loader       │
└────────┬────────┘     └──────────────┘
         │
         v
   ┌────────────┐
   │ For each   │
   │ model:     │
   └─────┬──────┘
         │
         v
   ┌─────────────────┐    ┌─────────────┐
   │ Model Wrapper   │───>│ Transcribe  │
   │ (load model)    │    │ All Samples │
   └─────────────────┘    └──────┬──────┘
                                 │
                                 v
                         ┌───────────────┐
                         │ Calculate     │
                         │ Metrics       │
                         └───────┬───────┘
                                 │
                                 v
                         ┌───────────────┐
                         │ Store Results │
                         └───────┬───────┘
                                 │
         ┌───────────────────────┘
         │
         v
   ┌─────────────┐
   │ Model       │
   │ Ranker      │
   └──────┬──────┘
          │
          v
   ┌─────────────┐
   │ Generate    │
   │ Reports     │
   └─────────────┘
```

## Extensibility

### Adding New Models

1. Add to `config.yaml`:

```yaml
models:
  my-model:
    name: 'huggingface/model-id'
    language: 'korean'
```

2. Run benchmark:

```bash
python -m src.main --model my-model
```

### Adding New Metrics

1. Implement in `src/metrics.py`:

```python
class CustomMetricCalculator:
    def calculate(self, refs, hyps):
        # Implementation
        return metric_value
```

2. Integrate in `benchmark.py`:

```python
custom_metric = CustomMetricCalculator()
result.custom_score = custom_metric.calculate(refs, preds)
```

### Adjusting Weights

Edit `src/ranking.py`:

```python
@dataclass
class RankingCriteria:
    cer_weight: float = 0.50  # Adjust as needed
    wer_weight: float = 0.20
    loanword_weight: float = 0.15
    speed_weight: float = 0.10
    ratio_weight: float = 0.05
```

## Testing Strategy

### Unit Tests

- `test_metrics.py`: Validates metric calculations
- `test_loanword_detector.py`: Tests loanword detection logic
- `test_ranking.py`: Verifies ranking algorithm

### Integration Tests

Run full benchmark on subset:

```yaml
benchmark:
  test_samples: 10 # Small subset for testing
```

## Performance Considerations

### Memory Management

- Models loaded one at a time
- Results stored incrementally
- Audio processed in batches (configurable)

### Speed Optimization

- GPU acceleration (when available)
- Batch processing
- Cached dataset downloads
- Parallel-ready architecture (future enhancement)

### Error Handling

- Model loading failures: skip and continue
- Transcription errors: logged, model marked as failed
- Dataset errors: fail fast with clear message

## Configuration Options

### Benchmark Settings

```yaml
benchmark:
  test_samples: 100 # Number of samples (null = all)
  batch_size: 8 # Batch size for processing
  device: 'cpu' # "cuda" or "cpu"
```

### Dataset Settings

```yaml
datasets:
  fleurs:
    subset: 'ko_kr' # Korean subset
    split: 'test' # train/validation/test
```

### Output Settings

```yaml
output:
  results_dir: 'results'
  save_predictions: true # Save detailed predictions
  generate_report: true # Generate markdown report
```

## Dependencies

### Core

- `transformers`: HuggingFace models
- `datasets`: FLEURS dataset loading
- `torch`: Deep learning framework
- `torchaudio`: Audio processing

### Metrics

- `jiwer`: CER/WER calculation
- `evaluate`: Evaluation utilities

### Data Processing

- `pandas`: Data manipulation
- `numpy`: Numerical operations

### I/O

- `pyyaml`: Configuration parsing
- `soundfile`: Audio file handling

## Future Enhancements

Potential improvements:

- Parallel model evaluation
- Real kitchen audio testing
- Noise augmentation
- Confusion matrix analysis
- Per-phoneme error analysis
- Interactive visualization dashboard
- Model quantization benchmarks
- Streaming inference support
