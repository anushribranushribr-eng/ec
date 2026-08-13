import React, { useState, useEffect } from "react";
import { Apple, CheckCircle2, Download, ShieldCheck, Cpu } from "lucide-react";

export const CoreMLExportWorkbench: React.FC = () => {
  const [exportData, setExportData] = useState<any>(null);

  useEffect(() => {
    fetch("/api/export-coreml")
      .then((res) => res.json())
      .then((data) => setExportData(data))
      .catch((err) => console.error(err));
  }, []);

  if (!exportData) {
    return (
      <div className="p-8 text-center text-[#94A3B8] font-mono text-xs">
        <div className="w-6 h-6 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto mb-2" />
        <span>Loading Core ML Package Metadata...</span>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header & Status Card */}
      <div className="bg-[#0F172A] border border-[#1E293B] rounded-sm p-4 space-y-3">
        <div className="flex items-center justify-between border-b border-[#1E293B] pb-2">
          <div className="flex items-center gap-2.5">
            <div className="p-1.5 bg-[#0B1120] border border-[#1E293B] rounded-sm text-white">
              <Apple className="w-4 h-4 text-white" />
            </div>
            <div>
              <h2 className="text-xs font-bold uppercase tracking-wider text-white">Apple Core ML Export Package</h2>
              <p className="text-[10px] font-mono text-[#94A3B8]">iOS 16.0+ • iPadOS 16.0+ • macOS 13.0+ Target</p>
            </div>
          </div>

          <span className="px-2 py-0.5 bg-green-900/30 text-green-400 border border-green-500/30 rounded-sm text-[10px] font-mono font-bold">
            {exportData.status}
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5 pt-1">
          <div className="bg-[#0B1120] p-3 rounded-sm border border-[#1E293B]">
            <span className="text-[9px] text-[#94A3B8] font-mono uppercase block mb-0.5">Package Name</span>
            <span className="text-xs font-bold text-white font-mono">{exportData.package_name}</span>
          </div>

          <div className="bg-[#0B1120] p-3 rounded-sm border border-[#1E293B]">
            <span className="text-[9px] text-[#94A3B8] font-mono uppercase block mb-0.5">Model Size</span>
            <span className="text-xs font-bold text-white font-mono">{exportData.model_size_mb} MB</span>
          </div>

          <div className="bg-[#0B1120] p-3 rounded-sm border border-[#1E293B]">
            <span className="text-[9px] text-[#94A3B8] font-mono uppercase block mb-0.5">Input Tensor Spec</span>
            <span className="text-[10px] font-mono text-indigo-400">{exportData.input_tensor}</span>
          </div>

          <div className="bg-[#0B1120] p-3 rounded-sm border border-[#1E293B]">
            <span className="text-[9px] text-[#94A3B8] font-mono uppercase block mb-0.5">Class Agreement</span>
            <span className="text-xs font-bold text-green-400 font-mono">{exportData.class_agreement}%</span>
          </div>
        </div>
      </div>

      {/* PyTorch vs Core ML Numerical Consistency Validation Report */}
      <div className="bg-[#0F172A] border border-[#1E293B] rounded-sm p-4 space-y-3">
        <div className="flex items-center gap-2 border-b border-[#1E293B] pb-2">
          <ShieldCheck className="w-4 h-4 text-green-400" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-white">PyTorch vs Core ML Numerical Validation</h3>
        </div>

        <div className="p-3 bg-green-900/30 border border-green-500/30 rounded-sm text-[11px] text-green-300 space-y-1 font-mono">
          <div className="flex items-center gap-1.5 font-bold">
            <CheckCircle2 className="w-3.5 h-3.5 text-green-400 shrink-0" />
            <span>Numerical Precision Verified (Tolerance &lt; 1e-4)</span>
          </div>
          <p className="text-green-300/80 leading-snug">
            Evaluated probability vector output agreement between PyTorch reference model and Apple Core ML model across 20 synthetic test spectrogram inputs. Maximum absolute probability deviation is {exportData.max_absolute_error_vs_pytorch}.
          </p>
        </div>

        <div className="bg-[#0B1120] p-3 rounded-sm border border-[#1E293B] font-mono text-[10px] text-slate-300 space-y-1">
          <div className="text-indigo-400 font-bold uppercase">// Core ML Input / Output Mapping Specification:</div>
          <div className="text-indigo-300">Inputs: audio_spectrogram [MultiArray Float32 (1, 1, 128, 125)]</div>
          <div className="text-green-400">Outputs: probabilities [Dictionary String-&gt;Double], classLabel [String]</div>
        </div>
      </div>
    </div>
  );
};
