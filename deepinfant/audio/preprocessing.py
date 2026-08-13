"""
Audio Preprocessing and Signal Conditioning.
Handles signal padding, cropping, silence trimming, and sliding window chunking.
"""

import numpy as np
import librosa
from typing import List, Tuple


def trim_silence(
    audio: np.ndarray,
    top_db: float = 30.0,
    frame_length: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """Removes excessive leading and trailing silence."""
    if len(audio) == 0:
        return audio
    trimmed, _ = librosa.effects.trim(
        audio, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )
    return trimmed


def pad_or_crop(
    audio: np.ndarray,
    target_length: int,
    mode: str = "center"
) -> np.ndarray:
    """
    Pads or crops 1D audio waveform to exact target_length samples.
    
    Args:
        audio: Input float32 waveform
        target_length: Target number of samples (e.g., 16000 * 4 = 64000)
        mode: 'center', 'start', or 'random'
    """
    current_length = len(audio)
    
    if current_length == target_length:
        return audio
        
    if current_length > target_length:
        # Crop
        if mode == "center":
            start = (current_length - target_length) // 2
        elif mode == "random":
            start = np.random.randint(0, current_length - target_length + 1)
        else:
            start = 0
        return audio[start : start + target_length]
    else:
        # Pad with zeros
        pad_len = target_length - current_length
        if mode == "center":
            left = pad_len // 2
            right = pad_len - left
            return np.pad(audio, (left, right), mode="constant")
        else:
            return np.pad(audio, (0, pad_len), mode="constant")


def extract_sliding_windows(
    audio: np.ndarray,
    sr: int = 16000,
    window_sec: float = 0.975,
    stride_sec: float = 0.25
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Splits long audio clip into overlapping sliding windows for inference.
    
    Returns:
        windows: List of 1D numpy float32 waveforms
        timestamps: Center timestamp (seconds) of each window
    """
    window_samples = int(window_sec * sr)
    stride_samples = int(stride_sec * sr)
    
    if len(audio) < window_samples:
        padded = pad_or_crop(audio, window_samples, mode="center")
        return [padded], [len(audio) / (2.0 * sr)]
        
    windows = []
    timestamps = []
    
    for start in range(0, len(audio) - window_samples + 1, stride_samples):
        end = start + window_samples
        chunk = audio[start:end]
        center_sec = (start + end) / (2.0 * sr)
        windows.append(chunk)
        timestamps.append(center_sec)
        
    return windows, timestamps
