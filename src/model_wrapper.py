"""
ASR Model Wrappers
API-based wrappers for OpenAI, Deepgram, and Gladia
"""
import os
import io
import time
import numpy as np
import soundfile as sf


def _audio_to_wav_bytes(audio: np.ndarray, sampling_rate: int) -> io.BytesIO:
    """Convert numpy audio array to in-memory WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, audio, sampling_rate, format="WAV")
    buf.seek(0)
    return buf


# ── OpenAI (whisper-1 or gpt-4o-transcribe) ───────────────────────────────────

class OpenAIWrapper:
    """
    Calls OpenAI audio transcription API.
    Supports both whisper-1 and gpt-4o-transcribe.
    Auto-detects language and handles code-switching.
    """

    def __init__(self, model_name: str, model: str = "gpt-4o-transcribe"):
        import openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Run: export OPENAI_API_KEY=your_key"
            )
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def transcribe(self, audio: np.ndarray, sampling_rate: int = 16000) -> str:
        buf = _audio_to_wav_bytes(audio, sampling_rate)
        buf.name = "audio.wav"
        response = self.client.audio.transcriptions.create(
            model=self.model,
            file=buf,
        )
        return response.text.strip()


# ── Deepgram Nova-3 ────────────────────────────────────────────────────────────

class DeepgramWrapper:
    """
    Calls Deepgram API with Nova-3 model.
    detect_language=true enables automatic language detection.
    Docs: https://developers.deepgram.com/docs/getting-started
    """

    def __init__(self, model_name: str, model: str = "nova-3"):
        api_key = os.environ.get("DEEPGRAM_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "DEEPGRAM_API_KEY not set. Get one at: https://console.deepgram.com\n"
                "Run: export DEEPGRAM_API_KEY=your_key"
            )
        self.model_name = model_name
        self.api_key = api_key
        self.model = model
        self.url = "https://api.deepgram.com/v1/listen"

    def transcribe(self, audio: np.ndarray, sampling_rate: int = 16000) -> str:
        import requests

        buf = _audio_to_wav_bytes(audio, sampling_rate)

        params = {
            "model": self.model,
            "detect_language": "true",
            "smart_format": "true",
        }
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "audio/wav",
        }
        response = requests.post(
            self.url,
            params=params,
            headers=headers,
            data=buf.read(),
        )
        response.raise_for_status()
        result = response.json()
        transcript = (
            result["results"]["channels"][0]["alternatives"][0]["transcript"]
        )
        return transcript.strip()


# ── Gladia ─────────────────────────────────────────────────────────────────────

class GladiaWrapper:
    """
    Calls Gladia transcription API.
    Supports 99 languages with automatic language detection.
    Docs: https://docs.gladia.io
    """

    def __init__(self, model_name: str):
        api_key = os.environ.get("GLADIA_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GLADIA_API_KEY not set. Get one at: https://app.gladia.io\n"
                "Run: export GLADIA_API_KEY=your_key"
            )
        self.model_name = model_name
        self.api_key = api_key
        self.upload_url = "https://api.gladia.io/v2/upload"
        self.transcription_url = "https://api.gladia.io/v2/pre-recorded"

    def transcribe(self, audio: np.ndarray, sampling_rate: int = 16000) -> str:
        import requests

        buf = _audio_to_wav_bytes(audio, sampling_rate)

        # Step 1: upload audio file (retry on 429)
        upload_resp = self._post_with_retry(
            self.upload_url,
            headers={"x-gladia-key": self.api_key},
            files={"audio": ("audio.wav", buf, "audio/wav")},
        )
        audio_url = upload_resp.json()["audio_url"]

        # Step 2: request transcription
        transcription_resp = self._post_with_retry(
            self.transcription_url,
            headers={
                "x-gladia-key": self.api_key,
                "Content-Type": "application/json",
            },
            json={
                "audio_url": audio_url,
                "detect_language": True,
                "enable_code_switching": True,
            },
        )
        result_url = transcription_resp.json()["result_url"]

        # Step 3: poll until done
        for _ in range(60):
            time.sleep(5)
            poll_resp = requests.get(
                result_url,
                headers={"x-gladia-key": self.api_key},
            )
            poll_resp.raise_for_status()
            data = poll_resp.json()
            if data.get("status") == "done":
                utterances = data["result"]["transcription"]["utterances"]
                return " ".join(u["text"] for u in utterances).strip()
            if data.get("status") == "error":
                raise RuntimeError(f"Gladia error: {data}")

        raise TimeoutError("Gladia transcription timed out after 5 minutes")

    def _post_with_retry(self, url: str, max_retries: int = 5, **kwargs):
        import requests
        delay = 10
        for attempt in range(max_retries):
            resp = requests.post(url, **kwargs)
            if resp.status_code == 429:
                print(f"  Gladia rate limit hit — waiting {delay}s before retry ({attempt+1}/{max_retries})")
                time.sleep(delay)
                delay *= 2  # exponential backoff
                continue
            resp.raise_for_status()
            return resp
        raise RuntimeError(f"Gladia rate limit exceeded after {max_retries} retries")


# ── Factory ────────────────────────────────────────────────────────────────────

def create_model_wrapper(model_name: str, model_config: dict, device: str = "cpu"):
    provider = model_config.get("provider")
    model_type = model_config.get("type", "api")

    if model_type == "api":
        if provider == "openai":
            return OpenAIWrapper(
                model_name=model_name,
                model=model_config.get("model", "gpt-4o-transcribe"),
            )
        elif provider == "deepgram":
            return DeepgramWrapper(
                model_name=model_name,
                model=model_config.get("model", "nova-3"),
            )
        elif provider == "gladia":
            return GladiaWrapper(model_name=model_name)
        else:
            raise ValueError(f"Unknown provider: {provider}. Use openai, deepgram, or gladia.")

    elif model_type == "local":
        return _load_local_model(model_name, model_config, device)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _load_local_model(model_name: str, model_config: dict, device: str):
    try:
        import torch
        from transformers import pipeline
    except ImportError:
        raise ImportError(
            "Local models require torch and transformers.\n"
            "Run: pip install torch transformers"
        )

    class LocalWrapper:
        def __init__(self):
            self.model_name = model_name
            torch_device = 0 if device == "cuda" and torch.cuda.is_available() else -1
            print(f"Loading {model_name} ({model_config['name']}) — requires model download...")
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model_config["name"],
                device=torch_device,
            )

        def transcribe(self, audio, sampling_rate=16000):
            result = self.pipe({"array": audio, "sampling_rate": sampling_rate})
            return result["text"].strip()

    return LocalWrapper()
