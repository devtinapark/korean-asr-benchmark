"""
Unified ASR Model Wrapper
Provides consistent interface for different ASR model architectures
"""
from typing import Dict, Optional, List
import torch
import numpy as np
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Wav2Vec2Processor, Wav2Vec2ForCTC,
    AutoProcessor, AutoModelForSpeechSeq2Seq, AutoModelForCTC,
    pipeline
)
import warnings
warnings.filterwarnings('ignore')


class ASRModelWrapper:
    """
    Unified wrapper for ASR models from different architectures
    """

    def __init__(
        self,
        model_name: str,
        model_id: str,
        language: Optional[str] = None,
        device: str = "cpu",
        use_pipeline: bool = True
    ):
        """
        Initialize ASR model wrapper

        Args:
            model_name: Display name for the model
            model_id: HuggingFace model ID
            language: Language code for the model
            device: Device to run model on (cpu/cuda)
            use_pipeline: Whether to use HF pipeline (simpler but less control)
        """
        self.model_name = model_name
        self.model_id = model_id
        self.language = language
        self.device = device
        self.use_pipeline = use_pipeline

        self.model = None
        self.processor = None
        self.pipe = None
        self.model_type = self._detect_model_type()

        print(f"Loading {model_name} ({model_id})...")
        self._load_model()

    def _detect_model_type(self) -> str:
        """
        Detect model architecture from model_id

        Returns:
            Model type string
        """
        model_id_lower = self.model_id.lower()

        if "whisper" in model_id_lower:
            return "whisper"
        elif "wav2vec2" in model_id_lower:
            return "wav2vec2"
        elif "hubert" in model_id_lower:
            return "hubert"
        elif "mms" in model_id_lower:
            return "mms"
        elif "canary" in model_id_lower:
            return "canary"
        elif "silero" in model_id_lower:
            return "silero"
        else:
            return "auto"

    def _load_model(self):
        """Load model and processor based on detected type"""
        try:
            if self.use_pipeline:
                self._load_via_pipeline()
            else:
                self._load_via_transformers()
        except Exception as e:
            print(f"Warning: Failed to load {self.model_name}: {e}")
            print("Attempting fallback loading method...")
            try:
                if self.use_pipeline:
                    self._load_via_transformers()
                else:
                    self._load_via_pipeline()
            except Exception as e2:
                print(f"Error: Could not load {self.model_name}: {e2}")
                raise

    def _load_via_pipeline(self):
        """Load model using HuggingFace pipeline"""
        try:
            # Determine task type
            task = "automatic-speech-recognition"

            # Create pipeline
            self.pipe = pipeline(
                task,
                model=self.model_id,
                device=0 if self.device == "cuda" and torch.cuda.is_available() else -1
            )
        except Exception as e:
            raise RuntimeError(f"Pipeline loading failed: {e}")

    def _load_via_transformers(self):
        """Load model using transformers directly"""
        if self.model_type in ["whisper"]:
            self.processor = WhisperProcessor.from_pretrained(self.model_id)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id)

        elif self.model_type in ["wav2vec2", "hubert", "mms"]:
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id)

        else:
            # Auto-detect
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            try:
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_id)
            except:
                self.model = AutoModelForCTC.from_pretrained(self.model_id)

        # Move to device
        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model.eval()

    def transcribe(
        self,
        audio: np.ndarray,
        sampling_rate: int = 16000
    ) -> str:
        """
        Transcribe audio to text

        Args:
            audio: Audio array
            sampling_rate: Audio sampling rate

        Returns:
            Transcribed text
        """
        try:
            if self.pipe is not None:
                return self._transcribe_pipeline(audio, sampling_rate)
            else:
                return self._transcribe_transformers(audio, sampling_rate)
        except Exception as e:
            print(f"Transcription error for {self.model_name}: {e}")
            return ""

    def _transcribe_pipeline(
        self,
        audio: np.ndarray,
        sampling_rate: int
    ) -> str:
        """Transcribe using pipeline"""
        # Prepare input
        audio_input = {
            "array": audio,
            "sampling_rate": sampling_rate
        }

        # Configure generation
        generate_kwargs = {}
        if self.language and self.model_type == "whisper":
            generate_kwargs["language"] = self.language

        # Transcribe
        result = self.pipe(
            audio_input,
            generate_kwargs=generate_kwargs,
            return_timestamps=False
        )

        return result["text"].strip()

    def _transcribe_transformers(
        self,
        audio: np.ndarray,
        sampling_rate: int
    ) -> str:
        """Transcribe using transformers directly"""
        # Process input
        inputs = self.processor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )

        # Move to device
        if self.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            if self.model_type == "whisper":
                # For Whisper, use forced decoder ids for language
                forced_decoder_ids = None
                if self.language:
                    forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                        language=self.language,
                        task="transcribe"
                    )

                generated_ids = self.model.generate(
                    inputs["input_features"],
                    forced_decoder_ids=forced_decoder_ids
                )
                transcription = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]

            else:
                # For CTC models (Wav2Vec2, Hubert, MMS)
                logits = self.model(**inputs).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)[0]

        return transcription.strip()

    def batch_transcribe(
        self,
        audio_list: List[np.ndarray],
        sampling_rate: int = 16000,
        batch_size: int = 8
    ) -> List[str]:
        """
        Transcribe multiple audio samples in batches

        Args:
            audio_list: List of audio arrays
            sampling_rate: Audio sampling rate
            batch_size: Batch size for processing

        Returns:
            List of transcriptions
        """
        transcriptions = []

        for i in range(0, len(audio_list), batch_size):
            batch = audio_list[i:i + batch_size]
            batch_transcriptions = [
                self.transcribe(audio, sampling_rate)
                for audio in batch
            ]
            transcriptions.extend(batch_transcriptions)

        return transcriptions

    def get_info(self) -> Dict:
        """
        Get model information

        Returns:
            Dictionary with model metadata
        """
        return {
            "name": self.model_name,
            "model_id": self.model_id,
            "type": self.model_type,
            "language": self.language,
            "device": self.device
        }
