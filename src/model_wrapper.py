"""
ASR Model Wrappers
Unified interface for API-based and local ASR models
"""
from typing import Optional
import os
import io
import numpy as np
import soundfile as sf


# ── Local HuggingFace wrapper ──────────────────────────────────────────────────

class LocalASRWrapper:
    """Runs a HuggingFace ASR model locally via transformers pipeline"""

    def __init__(self, model_name: str, model_id: str, language: Optional[str], device: str):
        import torch
        from transformers import pipeline

        self.model_name = model_name
        torch_device = 0 if device == "cuda" and torch.cuda.is_available() else -1

        print(f"Loading {model_name} ({model_id}) — this requires a model download...")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=torch_device,
        )
        self.language = language
        self.model_id = model_id

    def transcribe(self, audio: np.ndarray, sampling_rate: int = 16000) -> str:
        generate_kwargs = {}
        if self.language and "whisper" in self.model_id.lower():
            generate_kwargs["language"] = self.language

        result = self.pipe(
            {"array": audio, "sampling_rate": sampling_rate},
            generate_kwargs=generate_kwargs,
        )
        return result["text"].strip()


# ── OpenAI Whisper API ─────────────────────────────────────────────────────────

class OpenAIWhisperWrapper:
    """Calls the OpenAI Whisper API — no local model download needed"""

    def __init__(self, model_name: str, model: str = "whisper-1", language: str = "ko"):
        import openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY=your_key")

        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.language = language

    def transcribe(self, audio: np.ndarray, sampling_rate: int = 16000) -> str:
        # Write audio to an in-memory WAV buffer
        buf = io.BytesIO()
        sf.write(buf, audio, sampling_rate, format="WAV")
        buf.seek(0)
        buf.name = "audio.wav"  # OpenAI SDK uses the filename to detect format

        response = self.client.audio.transcriptions.create(
            model=self.model,
            file=buf,
            language=self.language,
        )
        return response.text.strip()


# ── Google Cloud Speech-to-Text ────────────────────────────────────────────────

class GoogleSpeechWrapper:
    """Calls Google Cloud Speech-to-Text API"""

    def __init__(self, model_name: str, language: str = "ko-KR"):
        from google.cloud import speech
        creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not creds:
            raise EnvironmentError(
                "GOOGLE_APPLICATION_CREDENTIALS is not set.\n"
                "Run: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json"
            )

        self.model_name = model_name
        self.client = speech.SpeechClient()
        self.language = language

    def transcribe(self, audio: np.ndarray, sampling_rate: int = 16000) -> str:
        from google.cloud import speech

        buf = io.BytesIO()
        sf.write(buf, audio, sampling_rate, format="WAV")
        content = buf.getvalue()

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sampling_rate,
            language_code=self.language,
        )
        audio_obj = speech.RecognitionAudio(content=content)
        response = self.client.recognize(config=config, audio=audio_obj)

        if not response.results:
            return ""
        return response.results[0].alternatives[0].transcript.strip()


# ── Naver Clova Speech ─────────────────────────────────────────────────────────

class ClovaSpeechWrapper:
    """
    Calls Naver Clova Speech API.
    Naver's Korean-native ASR — strong on colloquial Korean and Konglish.
    Docs: https://api.ncloud-docs.com/docs/ai-application-service-clovaspeech
    """

    def __init__(self, model_name: str, language: str = "Kor"):
        import requests  # noqa: F401 — just verify it's installed
        client_id = os.environ.get("NAVER_CLIENT_ID")
        client_secret = os.environ.get("NAVER_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise EnvironmentError(
                "NAVER_CLIENT_ID and NAVER_CLIENT_SECRET are not set.\n"
                "Get credentials at: https://www.ncloud.com/product/aiService/clovaSpeech"
            )

        self.model_name = model_name
        self.client_id = client_id
        self.client_secret = client_secret
        self.language = language
        self.endpoint = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt"

    def transcribe(self, audio: np.ndarray, sampling_rate: int = 16000) -> str:
        import requests

        buf = io.BytesIO()
        sf.write(buf, audio, sampling_rate, format="WAV")
        buf.seek(0)

        headers = {
            "X-NCP-APIGW-API-KEY-ID": self.client_id,
            "X-NCP-APIGW-API-KEY": self.client_secret,
            "Content-Type": "application/octet-stream",
        }
        response = requests.post(
            f"{self.endpoint}?lang={self.language}",
            headers=headers,
            data=buf.read(),
        )
        response.raise_for_status()
        return response.json().get("text", "").strip()


# ── Factory ────────────────────────────────────────────────────────────────────

def create_model_wrapper(model_name: str, model_config: dict, device: str = "cpu"):
    """
    Returns the right wrapper based on config type field.

    model_config must have:
      type: "api" | "local"
      provider: "openai" | "google" | "naver"   (when type=api)
      name: "org/model-id"                        (when type=local)
    """
    model_type = model_config.get("type", "local")

    if model_type == "api":
        provider = model_config["provider"]
        if provider == "openai":
            return OpenAIWhisperWrapper(
                model_name=model_name,
                model=model_config.get("model", "whisper-1"),
                language=model_config.get("language", "ko"),
            )
        elif provider == "google":
            return GoogleSpeechWrapper(
                model_name=model_name,
                language=model_config.get("language", "ko-KR"),
            )
        elif provider == "naver":
            return ClovaSpeechWrapper(
                model_name=model_name,
                language=model_config.get("language", "Kor"),
            )
        else:
            raise ValueError(f"Unknown API provider: {provider}")

    elif model_type == "local":
        return LocalASRWrapper(
            model_name=model_name,
            model_id=model_config["name"],
            language=model_config.get("language"),
            device=device,
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'api' or 'local'.")
