"""
PyTorch Dataset Implementation for DeepInfant V2.
Loads audio files, applies optional training augmentation, extracts 128-bin Log-Mel Spectrogram,
and encodes class targets.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple

from deepinfant.audio.loader import load_audio
from deepinfant.audio.preprocessing import pad_or_crop, trim_silence
from deepinfant.audio.augmentation import pitch_shift, time_stretch, add_background_noise, apply_spec_augment
from deepinfant.audio.features import compute_mel_spectrogram

CLASSES = [
    "belly_pain",
    "burping",
    "cold_hot",
    "discomfort",
    "hungry",
    "lonely",
    "scared",
    "tired",
    "unknown"
]

CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASSES)}


class InfantCryDataset(Dataset):
    """
    Infant Cry PyTorch Dataset.
    Loads audio, resamples to 16 kHz mono, applies training augmentations,
    computes 128-bin Log-Mel spectrogram, and outputs float32 torch tensors [1, n_mels, time_steps].
    """
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        audio_dir: Optional[str] = None,
        target_sr: int = 16000,
        duration_sec: float = 4.0,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        is_training: bool = False,
        augment_config: Optional[Dict] = None
    ):
        self.df = dataframe.copy().reset_index(drop=True)
        self.audio_dir = audio_dir
        self.target_sr = target_sr
        self.target_samples = int(target_sr * duration_sec)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.is_training = is_training
        self.augment_config = augment_config or {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # Determine file path
        if "file_path" in row and os.path.exists(str(row["file_path"])):
            file_path = str(row["file_path"])
        elif self.audio_dir and "filename" in row:
            file_path = os.path.join(self.audio_dir, str(row["filename"]))
        else:
            file_path = str(row.get("filename", ""))

        label_str = str(row.get("label", "unknown")).lower()
        class_idx = CLASS_TO_IDX.get(label_str, CLASS_TO_IDX["unknown"])
        
        # Load audio waveform
        try:
            audio, _ = load_audio(file_path, target_sr=self.target_sr, mono=True, normalize=True)
        except Exception:
            # Fallback to zero synthetic waveform on missing or corrupt file
            audio = np.zeros(self.target_samples, dtype=np.float32)

        # Apply waveform augmentations during training ONLY
        if self.is_training:
            if self.augment_config.get("pitch_shift", True) and np.random.rand() > 0.5:
                steps = np.random.uniform(-2.0, 2.0)
                audio = pitch_shift(audio, sr=self.target_sr, n_steps=steps)
                
            if self.augment_config.get("time_stretch", True) and np.random.rand() > 0.5:
                rate = np.random.uniform(0.9, 1.1)
                audio = time_stretch(audio, rate=rate)
                
            if self.augment_config.get("add_noise", True) and np.random.rand() > 0.5:
                audio = add_background_noise(audio, noise_factor=0.005)

        # Pad or crop to target_samples
        audio = pad_or_crop(audio, self.target_samples, mode="random" if self.is_training else "center")

        # Compute 128-bin Log-Mel Spectrogram
        log_mel = compute_mel_spectrogram(
            audio,
            sr=self.target_sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            log_scale=True
        ) # Shape: [n_mels, time_steps]

        # SpecAugment on spectrogram during training ONLY
        if self.is_training and self.augment_config.get("spec_augment", True) and np.random.rand() > 0.5:
            log_mel = apply_spec_augment(
                log_mel,
                freq_mask_param=self.augment_config.get("freq_mask", 15),
                time_mask_param=self.augment_config.get("time_mask", 35)
            )

        # Add channel dimension: [1, n_mels, time_steps]
        spectrogram_tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)
        target_tensor = torch.tensor(class_idx, dtype=torch.long)

        return spectrogram_tensor, target_tensor
