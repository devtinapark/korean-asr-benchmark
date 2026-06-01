# Multilingual ASR Benchmark — Kitchen Edition

Compares **OpenAI gpt-4o-transcribe vs Deepgram Nova-3** on real kitchen audio clips containing English, Korean, and code-switched speech. Tests clean vs. background noise.

No large dataset downloads — runs on your own audio files.

**[View live benchmark report →](https://devtinapark.github.io/korean-asr-benchmark)**

---

## Models Compared

Both models were selected because they auto-detect language and handle code-switching within a single clip — no language hint required.

| #   | Model               | Provider     | Why selected                                                                                 |
| --- | ------------------- | ------------ | -------------------------------------------------------------------------------------------- |
| 1   | `gpt-4o-transcribe` | OpenAI API   | Latest transcription model; auto-detects language; strong on Korean + English code-switching |
| 2   | Nova-3              | Deepgram API | Fast, `detect_language=true`; competitive accuracy; lowest cost per minute                   |

**Why other models were excluded:**

| Model                       | Reason                                                       |
| --------------------------- | ------------------------------------------------------------ |
| Google Cloud Speech-to-Text | Requires fixed language code per request — no auto-detection |
| Azure Speech                | Auto-detection requires a separate pre-pass                  |
| Naver Clova Speech          | Korean-only                                                  |
| Gladia                      | Free tier rate limits too restrictive for batch evaluation   |

---

## Sources / Leaderboard Attribution

For blog post citations — this comparison is based on:

1. **OpenAI Whisper paper** (Radford et al., 2022) — "Robust Speech Recognition via Large-Scale Weak Supervision." Table 8 shows per-language CER including Korean. [arxiv.org/abs/2212.04356](https://arxiv.org/abs/2212.04356)

2. **Papers With Code — Speech Recognition (Korean)** — Community leaderboard. [paperswithcode.com/task/speech-recognition](https://paperswithcode.com/task/speech-recognition)

3. **KsponSpeech benchmark** — Ko et al., 2020: "KsponSpeech: Korean Spontaneous Speech Corpus for Automatic Speech Recognition." [MDPI Applied Sciences](https://www.mdpi.com/2076-3417/10/19/6936)

4. **HuggingFace Open ASR Leaderboard** — Open model comparisons including Korean. [huggingface.co/spaces/hf-audio/open_asr_leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)

> Rankings reflect the state of multilingual ASR as of mid-2025. Verify against current leaderboards before publication.

---

## Quick Start

### 1. Install dependencies

```bash
conda create -n korean-asr python=3.11
conda activate korean-asr
pip install -r requirements.txt
```

Or with venv:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Also requires ffmpeg for MP3 support:

```bash
brew install ffmpeg
```

### 2. Add your audio clips

Place audio files in `kitchen_samples/audio/` and fill in `kitchen_samples/metadata.json`:

```json
[
  {
    "id": "001",
    "audio_file": "001_boil_water.wav",
    "transcript": "물이 끓고 있어요. 불 좀 줄여줘.",
    "noise": false
  }
]
```

Supported formats: `.wav`, `.mp3`, `.flac`, `.m4a`

### 3. Set API keys

```bash
export OPENAI_API_KEY=your_key      # platform.openai.com
export DEEPGRAM_API_KEY=your_key    # console.deepgram.com
```

### 4. Run

```bash
# Compare both models
python -m src.main

# Test one model
python -m src.main --model openai-gpt4o-transcribe
python -m src.main --model deepgram-nova-3

# List configured models
python -m src.main --list-models
```

---

## Output

```
results/
├── benchmark_report.html  # Visual comparison report — open in browser
├── benchmark_report.md    # Markdown report
├── results.csv
├── results.json
└── predictions/
    ├── openai-gpt4o-transcribe_predictions.csv
    └── deepgram-nova-3_predictions.csv
```

Open the HTML report:

```bash
open results/benchmark_report.html
```

---

## Metrics

| Metric                | What it measures                                                              | Good value      |
| --------------------- | ----------------------------------------------------------------------------- | --------------- |
| **CER**               | Character Error Rate — primary for Korean (spaces stripped before comparison) | < 0.05          |
| **WER**               | Word Error Rate                                                               | < 0.10          |
| **Loanword accuracy** | Code-switched terms like 오븐, 레시피, 간 맞추기                              | > 0.85          |
| **Noise delta**       | CER increase from clean → noisy clips                                         | Lower is better |
| **Latency**           | API response time per clip (excludes rate-limit pauses)                       | —               |
| **Cost**              | Estimated cost based on audio duration × price/min                            | —               |

Composite score: CER 55% + WER 30% + Loanword 15%. Speed excluded (measures API latency, not model quality).

> **Korean CER note:** Spaces are stripped before computing CER since Korean spacing (띄어쓰기) is inconsistent across models. This follows standard Korean ASR evaluation practice (KsponSpeech paper).

---

## Pricing (as of 2025-06)

| Model                    | Price/min | Free tier   |
| ------------------------ | --------- | ----------- |
| OpenAI gpt-4o-transcribe | $0.006    | No          |
| Deepgram Nova-3          | $0.0043   | $200 credit |

---

## Adding your own audio clips

Tips for recording:

- **Length**: 5–20 seconds per clip
- **Pairs**: record same content clean + with background noise to measure noise robustness
- **Content**: mix pure Korean, English, and code-switched phrases for multilingual testing
- **Set `"noise": true`** in metadata.json for noisy clips — the report uses this for noise impact analysis

---

## Contributing

PRs welcome:

- Additional audio clips (CC0 licensed)
- New API model wrappers
- Results from other languages (Spanish, Chinese)

## License

MIT
