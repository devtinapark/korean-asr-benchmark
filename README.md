# Korean ASR Benchmark — Kitchen Edition

Evaluates the top 5 ASR models for Korean kitchen voice use cases: casual speech, background noise, and English loanwords (Konglish like 오븐, 레시피, 타이머).

Tests against **your own audio clips** — no giant dataset downloads required.

---

## Why these 5 models?

These were selected from existing benchmarks and leaderboards, not tested from scratch here. See [Sources](#sources--leaderboard-attribution) below for blog post citation.

| #   | Model                                        | Type  | Why selected                                                                |
| --- | -------------------------------------------- | ----- | --------------------------------------------------------------------------- |
| 1   | OpenAI Whisper API (`whisper-1`)             | API   | Top multilingual CER in OpenAI's own published evals; no local download     |
| 2   | Google Cloud Speech-to-Text                  | API   | Consistently strong on Korean in independent benchmarks; handles noise well |
| 3   | Naver Clova Speech                           | API   | Korean-native model; best on colloquial Korean and Konglish                 |
| 4   | `openai/whisper-large-v3` (local)            | Local | Best open-source multilingual model; ~3GB download                          |
| 5   | `kresnik/wav2vec2-large-xlsr-korean` (local) | Local | Best publicly available Korean fine-tune on HuggingFace; ~1.3GB             |

---

## Sources / Leaderboard Attribution

1. **OpenAI Whisper paper** (Radford et al., 2022) — "Robust Speech Recognition via Large-Scale Weak Supervision." Table 8 shows per-language CER including Korean. Whisper large-v3 is the latest iteration. [arxiv.org/abs/2212.04356](https://arxiv.org/abs/2212.04356)

2. **Papers With Code — Speech Recognition (Korean)** — Community leaderboard tracking published results on Korean ASR benchmarks. [paperswithcode.com/task/speech-recognition](https://paperswithcode.com/task/speech-recognition)

3. **KsponSpeech benchmark** — The most widely used Korean spontaneous speech benchmark. Ko et al., 2020: "KsponSpeech: Korean Spontaneous Speech Corpus for Automatic Speech Recognition." Models are commonly compared here. [MDPI Applied Sciences](https://www.mdpi.com/2076-3417/10/19/6936)

4. **HuggingFace Open ASR Leaderboard** — Community benchmark comparing open models on multilingual tasks including Korean. [huggingface.co/spaces/hf-audio/open_asr_leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)

5. **Naver AI Lab** — Naver's research blog and papers on Korean ASR underpin Clova Speech's reputation as the strongest Korean-native commercial model. [ai.naver.com](https://ai.naver.com)

6. **`kresnik/wav2vec2-large-xlsr-korean`** — HuggingFace model card shows fine-tuning on AIHub Korean speech data; consistently cited in Korean NLP community comparisons.

> **Note:** These rankings reflect the state of Korean ASR as of mid-2025. Always verify against current leaderboards before publication.

---

## Quick Start

### 1. Install dependencies

```bash
conda create -n korean-asr python=3.11
conda activate korean-asr
pip install -r requirements.txt
```

### 2. Add your audio clips

Place `.wav` files in `kitchen_samples/audio/` and edit `kitchen_samples/metadata.json`:

```json
[
  {
    "id": "001",
    "audio_file": "001_boil_water.wav",
    "transcript": "물이 끓고 있어요. 불 좀 줄여줘.",
    "notes": "background: boiling water"
  }
]
```

Or use the simpler matching-filename approach — put `001.wav` in `audio/` and `001.txt` in `transcriptions/`.

### 3. Set API keys (for API models)

```bash
export OPENAI_API_KEY=your_key                              # OpenAI Whisper API
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/creds.json  # Google Speech
export NAVER_CLIENT_ID=your_id                              # Clova Speech
export NAVER_CLIENT_SECRET=your_secret
```

### 4. Run

```bash
# Test one model
python -m src.main --model openai-whisper-api

# Run all 5 models and rank them
python -m src.main

# List models in config
python -m src.main --list-models
```

---

## Output

Results saved to `results/`:

```
results/
├── benchmark_report.md    # Full ranked report
├── results.csv
├── results.json
└── predictions/           # Per-model transcript comparisons
```

---

## Metrics

| Metric                | What it measures                                          | Good value |
| --------------------- | --------------------------------------------------------- | ---------- |
| **CER**               | Character Error Rate — primary for Korean (agglutinative) | < 0.05     |
| **WER**               | Word Error Rate                                           | < 0.10     |
| **CER/WER ratio**     | Validates proper Korean morphology handling               | > 2.0x     |
| **Loanword accuracy** | Konglish words like 오븐, 레시피, 타이머                  | > 0.85     |

Composite score weights: CER 40%, WER 20%, Loanword 20%, Speed 10%, CER/WER ratio 10%.

---

## Adding your own audio clips

The benchmark is designed for your real-world recordings. Tips for kitchen clips:

- **Format**: 16kHz WAV mono (most models prefer this)
- **Length**: 5–20 seconds per clip
- **Content to cover**: pure Korean commands, Konglish terms (오븐, 믹서, 레시피), noisy background
- **Transcript format**: plain text, match what was actually said

---

## Local models (optional)

Local models require disk space and torch. To enable:

```bash
pip install torch>=2.4.0 torchaudio>=2.4.0 transformers>=4.40.0
```

Then run:

```bash
python -m src.main --model whisper-large-v3   # ~3GB download
python -m src.main --model wav2vec2-korean    # ~1.3GB download
```

---

## Contributing

PRs welcome — especially:

- Additional audio clips (CC0 licensed)
- New API model wrappers
- Korean dialect samples

## License

MIT
