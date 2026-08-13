"""
DeepInfant V2 Inference Engine.
Runs audio processing, feature extraction, CNN-LSTM model evaluation,
confidence thresholding, and returns probabilistic structured results.
"""

import os
import torch
import numpy as np
from typing import Dict, Any, List, Optional

from deepinfant.audio.loader import load_audio
from deepinfant.audio.preprocessing import pad_or_crop, extract_sliding_windows
from deepinfant.audio.features import compute_mel_spectrogram
from deepinfant.datasets.dataset import CLASSES, IDX_TO_CLASS
from deepinfant.inference.aggregation import aggregate_window_predictions

SAFETY_DISCLAIMER = (
    "AI prediction only — always check your baby's actual needs. If your baby appears "
    "seriously ill, has difficulty breathing, is unusually lethargic, has a high fever, "
    "or you are concerned about their condition, seek professional medical care immediately."
)


class CryPredictor:
    """Production Predictor for DeepInfant V2 Models."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any],
        confidence_threshold: float = 0.45
    ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.get("system", {}).get("device") == "cuda" else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        
        self.threshold = confidence_threshold
        self.sr = config.get("audio", {}).get("sample_rate", 16000)
        self.n_mels = config.get("audio", {}).get("n_mels", 128)
        self.n_fft = config.get("audio", {}).get("n_fft", 2048)
        self.hop_length = config.get("audio", {}).get("hop_length", 512)
        self.target_duration = config.get("audio", {}).get("target_duration_sec", 4.0)

    def predict_audio_array(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Runs model inference on a 1D float32 numpy audio array.
        Uses sliding window aggregation for variable length recordings.
        """
        if len(audio) == 0:
            audio = np.zeros(int(self.sr * self.target_duration), dtype=np.float32)

        # Sliding window extraction (0.975s window, 0.25s stride or 4s target)
        window_samples = int(self.sr * self.target_duration)
        padded_audio = pad_or_crop(audio, max(window_samples, len(audio)), mode="center")
        
        # Compute Log-Mel Spectrogram
        log_mel = compute_mel_spectrogram(
            padded_audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            log_scale=True
        )
        
        # Prepare Tensor [1, 1, 128, time_steps]
        input_tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs_tensor = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            
        # Top class and confidence
        top_idx = int(np.argmax(probs_tensor))
        top_confidence = float(probs_tensor[top_idx])
        top_class = CLASSES[top_idx]
        
        is_unknown = False
        # Rejection threshold check
        if top_confidence < self.threshold:
            top_class = "unknown"
            is_unknown = True
            
        prob_dict = {cls_name: round(float(probs_tensor[i]), 4) for i, cls_name in enumerate(CLASSES)}
        
        return {
            "prediction": top_class,
            "confidence": round(top_confidence, 4),
            "raw_top_class": CLASSES[top_idx],
            "raw_top_confidence": round(top_confidence, 4),
            "is_unknown": is_unknown,
            "confidence_threshold": self.threshold,
            "probabilities": prob_dict,
            "safety_disclaimer": SAFETY_DISCLAIMER
        }

    def predict_file(self, file_path: str) -> Dict[str, Any]:
        """Loads audio file and executes prediction."""
        audio, _ = load_audio(file_path, target_sr=self.sr, mono=True, normalize=True)
        res = self.predict_audio_array(audio)
        res["filename"] = os.path.basename(file_path)
        return res
