"""
FLEURS Korean Dataset Loader
Loads and preprocesses audio data for ASR benchmarking
"""
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, Dataset
import torch
import torchaudio
import numpy as np
from dataclasses import dataclass


@dataclass
class AudioSample:
    """Container for audio sample with metadata"""
    audio: np.ndarray
    sampling_rate: int
    text: str
    id: str
    raw_transcription: str


class FleursKoreanLoader:
    """
    Loads FLEURS Korean dataset for ASR evaluation
    """

    def __init__(
        self,
        subset: str = "ko_kr",
        split: str = "test",
        num_samples: Optional[int] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize FLEURS Korean dataset loader

        Args:
            subset: Language subset (default: ko_kr)
            split: Dataset split (train/validation/test)
            num_samples: Number of samples to load (None = all)
            cache_dir: Directory to cache dataset
        """
        self.subset = subset
        self.split = split
        self.num_samples = num_samples
        self.cache_dir = cache_dir
        self.dataset = None

    def load(self) -> Dataset:
        """
        Load FLEURS dataset from HuggingFace

        Returns:
            Loaded dataset
        """
        print(f"Loading FLEURS dataset: {self.subset} ({self.split} split)...")

        self.dataset = load_dataset(
            "google/fleurs",
            self.subset,
            split=self.split,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )

        if self.num_samples is not None:
            self.dataset = self.dataset.select(range(min(self.num_samples, len(self.dataset))))

        print(f"Loaded {len(self.dataset)} samples")
        return self.dataset

    def get_sample(self, idx: int) -> AudioSample:
        """
        Get a single preprocessed sample

        Args:
            idx: Sample index

        Returns:
            AudioSample object
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        sample = self.dataset[idx]

        # Extract audio array and sampling rate
        audio_data = sample['audio']
        audio_array = np.array(audio_data['array'], dtype=np.float32)
        sampling_rate = audio_data['sampling_rate']

        # Get transcription
        transcription = sample['transcription']

        return AudioSample(
            audio=audio_array,
            sampling_rate=sampling_rate,
            text=transcription,
            id=str(sample.get('id', idx)),
            raw_transcription=sample.get('raw_transcription', transcription)
        )

    def __len__(self) -> int:
        """Return dataset size"""
        if self.dataset is None:
            return 0
        return len(self.dataset)

    def __iter__(self):
        """Iterate over all samples"""
        for idx in range(len(self)):
            yield self.get_sample(idx)


class AudioPreprocessor:
    """
    Preprocesses audio for different ASR models
    """

    @staticmethod
    def resample_audio(
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """
        Resample audio to target sampling rate

        Args:
            audio: Audio array
            orig_sr: Original sampling rate
            target_sr: Target sampling rate

        Returns:
            Resampled audio array
        """
        if orig_sr == target_sr:
            return audio

        # Convert to torch tensor for resampling
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        resampled = resampler(audio_tensor).squeeze(0).numpy()

        return resampled

    @staticmethod
    def normalize_audio(audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range

        Args:
            audio: Audio array

        Returns:
            Normalized audio array
        """
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / np.max(np.abs(audio))
        return audio

    @staticmethod
    def prepare_for_model(
        audio: np.ndarray,
        sampling_rate: int,
        target_sr: int = 16000,
        normalize: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Prepare audio for ASR model inference

        Args:
            audio: Audio array
            sampling_rate: Current sampling rate
            target_sr: Target sampling rate for model
            normalize: Whether to normalize audio

        Returns:
            Tuple of (processed_audio, sampling_rate)
        """
        # Resample if needed
        if sampling_rate != target_sr:
            audio = AudioPreprocessor.resample_audio(audio, sampling_rate, target_sr)
            sampling_rate = target_sr

        # Normalize if requested
        if normalize:
            audio = AudioPreprocessor.normalize_audio(audio)

        return audio, sampling_rate
