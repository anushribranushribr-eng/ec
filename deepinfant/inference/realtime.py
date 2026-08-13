"""
Real-Time Microphone Audio Stream & Rolling Buffer Inference Engine.
Maintains a rolling audio buffer, checks Voice/Cry Activity Detection (VAD),
and produces smooth real-time predictions.
"""

import numpy as np
from typing import Dict, Any, Optional
from deepinfant.inference.aggregation import TemporalPredictionSmoother
from deepinfant.inference.predictor import CryPredictor


class CryActivityDetector:
    """
    Voice / Cry Activity Detector (VAD).
    Filters out background silence, ambient hums, and low-energy adult speech.
    """
    
    def __init__(
        self,
        energy_threshold: float = 0.015,
        freq_min: float = 250.0,
        freq_max: float = 3500.0
    ):
        self.energy_threshold = energy_threshold
        self.freq_min = freq_min
        self.freq_max = freq_max

    def is_cry_present(self, audio: np.ndarray, sr: int = 16000) -> bool:
        """
        Calculates Root Mean Square (RMS) energy and spectral energy in the infant fundamental pitch range (250Hz - 3500Hz).
        """
        if len(audio) == 0:
            return False
            
        rms_energy = np.sqrt(np.mean(audio ** 2))
        if rms_energy < self.energy_threshold:
            return False
            
        # Basic spectral check
        fft_vals = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
        
        band_mask = (freqs >= self.freq_min) & (freqs <= self.freq_max)
        band_energy = np.sum(fft_vals[band_mask] ** 2)
        total_energy = np.sum(fft_vals ** 2) + 1e-8
        
        cry_spectral_ratio = band_energy / total_energy
        return bool(cry_spectral_ratio > 0.30)


class RealtimeInferencePipeline:
    """
    Real-time rolling audio inference manager.
    """
    
    def __init__(
        self,
        predictor: CryPredictor,
        buffer_duration_sec: float = 4.0,
        sr: int = 16000,
        smoothing_alpha: float = 0.35,
        enable_vad: bool = True
    ):
        self.predictor = predictor
        self.sr = sr
        self.buffer_size = int(buffer_duration_sec * sr)
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.smoother = TemporalPredictionSmoother(alpha=smoothing_alpha)
        self.vad = CryActivityDetector()
        self.enable_vad = enable_vad

    def push_audio_chunk(self, chunk: np.ndarray) -> Dict[str, Any]:
        """
        Appends new PCM float32 audio chunk to rolling buffer and executes inference.
        """
        chunk_len = len(chunk)
        if chunk_len >= self.buffer_size:
            self.buffer = chunk[-self.buffer_size:].astype(np.float32)
        else:
            self.buffer = np.roll(self.buffer, -chunk_len)
            self.buffer[-chunk_len:] = chunk.astype(np.float32)

        # Check VAD
        is_cry = self.vad.is_cry_present(self.buffer, sr=self.sr) if self.enable_vad else True
        
        if not is_cry:
            # If silence or background noise, decay predictions towards unknown
            uniform_probs = np.ones(9) / 9.0
            smoothed_probs = self.smoother.update(uniform_probs)
            return {
                "cry_detected": False,
                "prediction": "no_cry_detected",
                "confidence": 0.0,
                "probabilities": {cls_name: float(smoothed_probs[i]) for i, cls_name in enumerate(self.predictor.config.get("classes", []))},
                "status": "Listening... No infant cry detected."
            }

        # Predict raw probabilities
        result = self.predictor.predict_audio_array(self.buffer)
        raw_probs_list = [result["probabilities"][cls_name] for cls_name in self.predictor.config.get("classes", [])]
        
        # Apply temporal smoothing
        smoothed_probs = self.smoother.update(np.array(raw_probs_list))
        
        top_idx = int(np.argmax(smoothed_probs))
        top_confidence = float(smoothed_probs[top_idx])
        top_class = self.predictor.config.get("classes", [])[top_idx]
        
        if top_confidence < self.predictor.threshold:
            top_class = "unknown"
            
        result["prediction"] = top_class
        result["confidence"] = round(top_confidence, 4)
        result["cry_detected"] = True
        result["probabilities"] = {
            cls_name: round(float(smoothed_probs[i]), 4)
            for i, cls_name in enumerate(self.predictor.config.get("classes", []))
        }
        return result
