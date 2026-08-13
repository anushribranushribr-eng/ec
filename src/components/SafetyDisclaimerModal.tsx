import React from "react";
import { AlertTriangle, CheckCircle2, HeartPulse, X } from "lucide-react";

interface SafetyDisclaimerModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const SafetyDisclaimerModal: React.FC<SafetyDisclaimerModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-[#000000]/90 backdrop-blur-md">
      <div className="bg-[#1a1a1a] border border-[#3c3c3c] max-w-lg w-full p-6 text-[#f5f5f5] relative font-mono space-y-5">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-[#737373] hover:text-[#f5f5f5] p-1 transition"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-3 border-b border-[#3c3c3c] pb-4">
          <div className="p-2.5 bg-[#000000] border border-[#ff4f2b] text-[#ff4f2b]">
            <AlertTriangle className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-sm font-bold uppercase text-[#f5f5f5] tracking-wider">SAFETY NOTICE // NON_MEDICAL</h3>
            <p className="text-xs text-[#737373]">ASSISTIVE_INFANT_CRY_CLASSIFICATION_SYSTEM</p>
          </div>
        </div>

        <div className="space-y-4 text-xs leading-relaxed text-[#bfbfbf]">
          <div className="bg-[#000000] border border-[#ff4f2b] p-4 text-[#f5f5f5] text-xs leading-relaxed">
            "AI prediction only — always check your baby's actual needs. If your baby appears unwell, has difficulty breathing, or you are concerned about their health, seek medical care immediately."
          </div>

          <p className="text-xs">
            <strong className="text-[#f5f5f5] uppercase font-bold">DEEPINFANT_V2</strong> provides an experimental AI-based interpretation of infant vocalizations by comparing acoustic audio patterns to annotated datasets.
          </p>

          <ul className="space-y-2 text-xs text-[#bfbfbf]">
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-[#ff4f2b] shrink-0 mt-0.5" />
              <span>Predictions are probabilistic acoustic suggestions, not medical diagnoses.</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-[#ff4f2b] shrink-0 mt-0.5" />
              <span>Outputs must never replace parental intuition or professional pediatric advice.</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-[#ff4f2b] shrink-0 mt-0.5" />
              <span>Confidence threshold is set to 45% — ambiguous signatures are marked as 'uncertain'.</span>
            </li>
          </ul>
        </div>

        <div className="pt-4 border-t border-[#3c3c3c] flex justify-end">
          <button
            onClick={onClose}
            className="px-6 py-3 bg-[#ff4f2b] hover:bg-[#000000] text-[#000000] hover:text-[#ff4f2b] font-bold text-xs uppercase tracking-wider border border-[#ff4f2b] transition"
          >
            I UNDERSTAND & ACCEPT
          </button>
        </div>
      </div>
    </div>
  );
};
