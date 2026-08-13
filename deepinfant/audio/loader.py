"""
Audio Loading and Conversion Module.
Resamples to canonical 16,000 Hz, converts multi-channel audio to mono, and normalizes amplitude.
"""

import os
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional


def load_audio(
    file_path: str,
    target_sr: int = 16000,
    mono: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Loads audio file, converts to mono, resamples to target_sr (16 kHz), and normalizes amplitude.
    
    Args:
        file_path: Path to the audio file (.wav, .m4a, .caf, .3gp, etc.)
        target_sr: Target sample rate (default: 16000 Hz)
        mono: Force single channel mono output
        normalize: Apply peak amplitude normalization
        
    Returns:
        audio: 1D numpy array float32 audio waveform
        sr: Sample rate (16000)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found at: {file_path}")
        
    # Load audio using librosa
    audio, sr = librosa.load(file_path, sr=target_sr, mono=mono)
    
    # Ensure float32 format
    audio = audio.astype(np.float32)
    
    # Amplitude peak normalization to [-1.0, 1.0]
    if normalize and len(audio) > 0:
        max_val = np.max(np.abs(audio))
        if max_val > 1e-6:
            audio = audio / max_val
            
    return audio, target_sr


def save_audio(file_path: str, audio: np.ndarray, sr: int = 16000) -> None:
    """Saves float32 audio array to WAV file."""
    sf.write(file_path, audio, sr)
