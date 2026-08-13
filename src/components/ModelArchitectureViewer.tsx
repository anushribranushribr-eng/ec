import React, { useState } from "react";
import { Cpu, ArrowRight, Layers, Volume2, Music, CheckCircle2, Sliders } from "lucide-react";

export const ModelArchitectureViewer: React.FC = () => {
  const [selectedBackbone, setSelectedBackbone] = useState<string>("cnn_lstm");
  const [activeStep, setActiveStep] = useState<number>(3);

  const steps = [
    {
      step: 1,
      title: "Audio Acquisition & Conditioning",
      spec: "16,000 Hz • Mono PCM",
      description: "Converts stereo input to 16 kHz mono float32 waveform, trims excessive silence, and normalizes peak amplitude to [-1.0, 1.0]."
    },
    {
      step: 2,
      title: "STFT & 128-bin Mel-Spectrogram",
      spec: "n_fft=2048 • hop=512 • Hann",
      description: "Computes Short-Time Fourier Transform with 2048 FFT window, projects onto 128 Mel frequency filterbank, and applies log normalization log(mel + eps)."
    },
    {
      step: 3,
      title: "2D CNN Spatial Feature Extractor",
      spec: selectedBackbone === "cnn_lstm" ? "4 Conv2D Stages + BatchNorm + ReLU" : "2D ResNet-Style Residual Blocks",
      description: "Extracts local time-frequency acoustic patterns across 128 Mel frequency bins into a compressed feature sequence."
    },
    {
      step: 4,
      title: "Bidirectional LSTM Sequence Modeling",
      spec: "2 Layers • 128 Hidden Dim • Dropout 0.3",
      description: "Captures temporal cry cadence, rhythm, and pitch progression across overlapping sliding time steps."
    },
    {
      step: 5,
      title: "Softmax Classifier Head",
      spec: "9 Classes • Rejection Thresholding",
      description: "Dense classification layer producing a 9-class probability distribution. Rejects uncertain classifications (< 45% confidence) to 'unknown'."
    }
  ];

  return (
    <div className="space-y-4">
      {/* Header & Backbone Selector */}
      <div className="bg-[#0F172A] border border-[#1E293B] rounded-sm p-4 flex flex-col md:flex-row items-start md:items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 mb-0.5">
            <Cpu className="w-4 h-4 text-indigo-400" />
            <h2 className="text-xs font-bold uppercase tracking-wider text-white">DeepInfant V2 Model Pipeline</h2>
          </div>
          <p className="text-[10px] font-mono text-[#94A3B8]">Hybrid Convolutional Neural Network & Bidirectional Recurrent Sequence Modeling</p>
        </div>

        <div className="flex items-center gap-1 bg-[#0B1120] p-1 rounded-sm border border-[#1E293B] text-[11px] font-mono">
          <span className="text-[#94A3B8] px-1.5 uppercase text-[10px]">Backbone:</span>
          {[
            { id: "cnn_lstm", label: "CNN + BiLSTM" },
            { id: "resnet18_lstm", label: "ResNet18 + LSTM" },
            { id: "resnet34_lstm", label: "ResNet34 + LSTM" }
          ].map((item) => (
            <button
              key={item.id}
              onClick={() => setSelectedBackbone(item.id)}
              className={`px-2.5 py-1 rounded-sm font-bold transition ${
                selectedBackbone === item.id
                  ? "bg-indigo-600 text-white"
                  : "text-[#94A3B8] hover:text-white hover:bg-[#1E293B]"
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      {/* Visual Workflow Steps */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-2.5">
        {steps.map((item) => {
          const isActive = activeStep === item.step;
          return (
            <button
              key={item.step}
              onClick={() => setActiveStep(item.step)}
              className={`p-3 rounded-sm border text-left transition-all relative ${
                isActive
                  ? "bg-[#1E293B] border-indigo-500 text-white"
                  : "bg-[#0F172A] border-[#1E293B] text-[#94A3B8] hover:border-slate-700 hover:bg-[#1E293B]/40"
              }`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <span
                  className={`w-5 h-5 rounded-sm flex items-center justify-center text-[10px] font-mono font-bold ${
                    isActive ? "bg-indigo-600 text-white" : "bg-[#0B1120] text-[#94A3B8]"
                  }`}
                >
                  {item.step}
                </span>
                {isActive && <CheckCircle2 className="w-3.5 h-3.5 text-indigo-400" />}
              </div>

              <h3 className="text-[11px] font-bold text-white mb-0.5">{item.title}</h3>
              <p className="text-[9px] text-indigo-400 font-mono mb-1">{item.spec}</p>
            </button>
          );
        })}
      </div>

      {/* Active Step Detailed Inspector */}
      {activeStep && (
        <div className="bg-[#0F172A] border border-[#1E293B] rounded-sm p-4 space-y-3">
          <div className="flex items-center gap-2.5 border-b border-[#1E293B] pb-2">
            <div className="px-2 py-0.5 bg-indigo-900/40 border border-indigo-500/30 rounded-sm text-indigo-300 font-mono font-bold text-xs">
              STEP {activeStep}
            </div>
            <div>
              <h3 className="text-xs font-bold text-white uppercase">{steps[activeStep - 1].title}</h3>
              <p className="text-[10px] font-mono text-indigo-400">{steps[activeStep - 1].spec}</p>
            </div>
          </div>

          <p className="text-xs text-slate-300 leading-snug">
            {steps[activeStep - 1].description}
          </p>

          <div className="bg-[#0B1120] p-3 rounded-sm border border-[#1E293B] font-mono text-[10px] text-slate-300 space-y-0.5">
            <div className="text-indigo-400 font-bold uppercase mb-1">// Mathematical / Tensor Specification:</div>
            {activeStep === 1 && <div>x_waveform = resample_16k(mono_audio) // Shape: [1, 64000]</div>}
            {activeStep === 2 && <div>X_stft = STFT(x, n_fft=2048, hop=512) // Log-Mel: [1, 1, 128, 125]</div>}
            {activeStep === 3 && <div>features_2d = Conv2D_Backbone(X_stft) // Compressed: [batch, 125, 4096]</div>}
            {activeStep === 4 && <div>h_seq = BiLSTM(features_2d) // Hidden Sequence: [batch, 125, 256]</div>}
            {activeStep === 5 && <div>probs = Softmax(Dense(Pooling(h_seq))) // Output: [batch, 9]</div>}
          </div>
        </div>
      )}
    </div>
  );
};
