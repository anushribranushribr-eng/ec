import React, { useState } from "react";
import { Folder, FileCode, Check, Sparkles } from "lucide-react";

const PYTHON_CODEBASE = [
  {
    path: "configs/default.yaml",
    category: "Configuration",
    content: `# DeepInfant V2 Default Configuration
system:
  project_name: "DeepInfant V2"
  device: "cuda"
  seed: 42

audio:
  sample_rate: 16000
  n_fft: 2048
  hop_length: 512
  n_mels: 128

model:
  backbone: "cnn_lstm"
  num_classes: 9

training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  loss_function: "weighted_cross_entropy"`
  },
  {
    path: "deepinfant/audio/features.py",
    category: "Audio Processing",
    content: `import numpy as np
import librosa

def compute_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    log_scale: bool = True
) -> np.ndarray:
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0
    )
    if log_scale:
        log_mel = np.log(mel_spec + 1e-6)
        return ((log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-8)).astype(np.float32)
    return mel_spec.astype(np.float32)`
  },
  {
    path: "deepinfant/models/cnn_lstm.py",
    category: "Core Model Architecture",
    content: `import torch
import torch.nn as nn
from deepinfant.models.cnn import CNNFeatureExtractor

class DeepInfantCNNLSTM(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 9, lstm_hidden_dim: int = 128):
        super().__init__()
        self.cnn = CNNFeatureExtractor(in_channels=in_channels)
        self.projection = nn.Sequential(nn.Linear(256 * 16, 256), nn.ReLU(), nn.Dropout(0.3))
        self.lstm = nn.LSTM(256, lstm_hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn(x)
        proj = self.projection(cnn_feat)
        lstm_out, _ = self.lstm(proj)
        pooled = torch.mean(lstm_out, dim=1)
        return self.fc(pooled)`
  },
  {
    path: "deepinfant/datasets/splitting.py",
    category: "Data Integrity",
    content: `import numpy as np
import pandas as pd

def subject_level_split(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15, seed: int = 42):
    """CRITICAL: Prevents data leakage by ensuring NO baby_id occurs in more than one split!"""
    np.random.seed(seed)
    unique_subjects = df["baby_id"].unique()
    np.random.shuffle(unique_subjects)
    
    num_train = int(np.round(train_ratio * len(unique_subjects)))
    num_val = int(np.round(0.15 * len(unique_subjects)))
    
    train_subjects = set(unique_subjects[:num_train])
    val_subjects = set(unique_subjects[num_train : num_train + num_val])
    test_subjects = set(unique_subjects[num_train + num_val :])
    
    return (
        df[df["baby_id"].isin(train_subjects)],
        df[df["baby_id"].isin(val_subjects)],
        df[df["baby_id"].isin(test_subjects)]
    )`
  },
  {
    path: "deepinfant/export/coreml.py",
    category: "Model Deployment",
    content: `import torch
import numpy as np

def export_pytorch_to_coreml(pytorch_model: torch.nn.Module, output_path: str = "DeepInfant_V2.mlpackage"):
    pytorch_model.eval()
    dummy_input = torch.randn(1, 1, 128, 125)
    traced = torch.jit.trace(pytorch_model, dummy_input)
    # Convert TorchScript to Core ML
    import coremltools as ct
    mlmodel = ct.convert(traced, inputs=[ct.TensorType(name="audio_spectrogram", shape=(1, 1, 128, 125))])
    mlmodel.save(output_path)
    return output_path`
  }
];

export const CodebaseExplorer: React.FC = () => {
  const [selectedIdx, setSelectedIdx] = useState<number>(2);
  const [copied, setCopied] = useState<boolean>(false);

  const active = PYTHON_CODEBASE[selectedIdx];

  const handleCopy = () => {
    navigator.clipboard.writeText(active.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-4">
      <div className="bg-[#0F172A] border border-[#1E293B] rounded-sm p-4 flex items-center justify-between">
        <div>
          <h2 className="text-xs font-bold uppercase tracking-wider text-white flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-indigo-400" />
            <span>Python Machine Learning Architecture Explorer</span>
          </h2>
          <p className="text-[10px] font-mono text-[#94A3B8]">DeepInfant V2 PyTorch, librosa, STFT, CNN-BiLSTM & Core ML package</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left Tree */}
        <div className="lg:col-span-4 bg-[#0F172A] border border-[#1E293B] rounded-sm p-3 space-y-1.5">
          <span className="text-[10px] font-mono font-bold text-[#94A3B8] uppercase tracking-wider block mb-1 px-1">
            Python Modules
          </span>

          {PYTHON_CODEBASE.map((item, idx) => (
            <button
              key={item.path}
              onClick={() => setSelectedIdx(idx)}
              className={`w-full p-2.5 rounded-sm text-left border transition ${
                selectedIdx === idx
                  ? "bg-[#1E293B] border-indigo-500 text-white"
                  : "bg-[#0B1120] border-[#1E293B] hover:border-slate-700 text-[#94A3B8]"
              }`}
            >
              <div className="flex items-center gap-1.5 mb-0.5">
                <FileCode className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
                <span className="text-xs font-mono font-bold truncate">{item.path.split("/").pop()}</span>
              </div>
              <span className="text-[10px] text-[#94A3B8] font-mono block">{item.path}</span>
            </button>
          ))}
        </div>

        {/* Right Code */}
        <div className="lg:col-span-8 bg-[#0F172A] border border-[#1E293B] rounded-sm p-4 space-y-2.5">
          <div className="flex items-center justify-between border-b border-[#1E293B] pb-2">
            <div>
              <h3 className="text-xs font-bold text-white font-mono">{active.path}</h3>
              <span className="text-[10px] font-mono text-indigo-400 font-bold uppercase">{active.category}</span>
            </div>

            <button
              onClick={handleCopy}
              className="px-2.5 py-1 text-[10px] font-mono uppercase font-bold bg-[#0B1120] hover:bg-[#1E293B] text-slate-200 border border-[#1E293B] rounded-sm transition flex items-center gap-1"
            >
              {copied ? <Check className="w-3 h-3 text-green-400" /> : null}
              <span>{copied ? "Copied" : "Copy Source"}</span>
            </button>
          </div>

          <pre className="bg-[#0B1120] p-3 rounded-sm border border-[#1E293B] text-[11px] font-mono text-slate-200 overflow-x-auto max-h-[500px]">
            <code>{active.content}</code>
          </pre>
        </div>
      </div>
    </div>
  );
};
