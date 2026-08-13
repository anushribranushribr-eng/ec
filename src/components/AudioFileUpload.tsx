import React, { useState } from "react";
import { UploadCloud, Play, FileAudio, CheckCircle2, ShieldAlert, Sparkles } from "lucide-react";
import { CryClass, PredictionResult, SampleCry } from "../types";
import { CRY_CLASS_METADATA } from "../utils/audioProcessor";

const PRELOADED_SAMPLES: SampleCry[] = [
  {
    id: "s1",
    label: "hungry",
    displayName: "Sample 1: Hungry Cry",
    description: "Rhythmic, repetitive 'neh'-like sound pattern.",
    filename: "0D1AD73E-4C5E-45F3-85C4-9A3CB71E8856-1430742197-1.0-m-04-hu.caf",
    duration: "4.0s",
    gender: "male",
    ageMonths: 4
  },
  {
    id: "s2",
    label: "belly_pain",
    displayName: "Sample 2: Belly Pain / Gas",
    description: "High-pitched intense cry with leg contraction pauses.",
    filename: "9F3C1A2E-8B7C-42D1-90E3-112233445566-1430748800-1.0-f-02-bp.caf",
    duration: "3.8s",
    gender: "female",
    ageMonths: 2
  },
  {
    id: "s3",
    label: "burping",
    displayName: "Sample 3: Needs Burping",
    description: "Short guttural 'eh' sound from chest tightness.",
    filename: "3A2B1C0D-4E5F-6789-0011-223344556677-1430750000-1.0-m-03-bu.caf",
    duration: "4.2s",
    gender: "male",
    ageMonths: 3
  },
  {
    id: "s4",
    label: "discomfort",
    displayName: "Sample 4: General Discomfort",
    description: "Fretful nasal 'eairh' sound from wet diaper or posture.",
    filename: "77889900-1122-3344-5566-778899aabbcc-1430751000-1.0-f-01-dc.caf",
    duration: "3.5s",
    gender: "female",
    ageMonths: 1
  },
  {
    id: "s5",
    label: "tired",
    displayName: "Sample 5: Sleepy / Tired",
    description: "Low-pitch yawning 'owh' vocalization.",
    filename: "11223344-5566-7788-9900-aabbccddeeff-1430752000-1.0-m-05-ti.caf",
    duration: "4.1s",
    gender: "male",
    ageMonths: 5
  },
  {
    id: "s6",
    label: "scared",
    displayName: "Sample 6: Startled / Scared",
    description: "Sudden high intensity cry following surprise noise.",
    filename: "aabbccdd-eeff-0011-2233-445566778899-1430753000-1.0-f-03-sc.caf",
    duration: "3.9s",
    gender: "female",
    ageMonths: 3
  }
];

export const AudioFileUpload: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [activeSample, setActiveSample] = useState<SampleCry | null>(PRELOADED_SAMPLES[0]);

  const [analysisResult, setAnalysisResult] = useState<PredictionResult | null>({
    prediction: "hungry",
    confidence: 0.724,
    rawTopClass: "hungry",
    rawTopConfidence: 0.724,
    isUnknown: false,
    confidenceThreshold: 0.45,
    probabilities: {
      hungry: 0.724,
      discomfort: 0.082,
      tired: 0.061,
      burping: 0.045,
      belly_pain: 0.031,
      unknown: 0.027,
      lonely: 0.014,
      scared: 0.01,
      cold_hot: 0.006
    },
    safetyDisclaimer: "AI prediction only — always check your baby's actual needs."
  });

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setActiveSample(null);
      runPredictionForFile(file);
    }
  };

  const handleSampleSelect = (sample: SampleCry) => {
    setActiveSample(sample);
    setSelectedFile(null);
    runPredictionForFile(sample.filename, sample.label);
  };

  const runPredictionForFile = (fileOrFilename: File | string, requestedType?: string) => {
    setIsLoading(true);

    if (fileOrFilename instanceof File) {
      const formData = new FormData();
      formData.append("audio", fileOrFilename);
      formData.append("threshold", "0.45");
      if (requestedType) {
        formData.append("cry_type", requestedType);
      }

      fetch("/api/predict", {
        method: "POST",
        body: formData
      })
        .then((res) => res.json())
        .then((data) => {
          setAnalysisResult(data);
        })
        .catch((err) => console.error(err))
        .finally(() => setIsLoading(false));
    } else {
      fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filename: fileOrFilename,
          cry_type: requestedType,
          threshold: 0.45
        })
      })
        .then((res) => res.json())
        .then((data) => {
          setAnalysisResult(data);
        })
        .catch((err) => console.error(err))
        .finally(() => setIsLoading(false));
    }
  };

  const topClass = analysisResult?.prediction || "unknown";
  const topMeta = CRY_CLASS_METADATA[topClass as CryClass] || CRY_CLASS_METADATA.unknown;

  return (
    <div className="space-y-6">
      {/* Preloaded Audio Samples Grid */}
      <div className="bg-[#1a1a1a] border border-[#3c3c3c] p-6 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-[#3c3c3c] pb-4 gap-2">
          <div>
            <h2 className="font-sans font-light text-xl sm:text-2xl text-[#f5f5f5] uppercase tracking-tight flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-[#ff4f2b]" />
              <span>Preloaded Acoustic Samples</span>
            </h2>
            <p className="font-mono text-xs text-[#bfbfbf] mt-1">
              Select an annotated audio sample to execute STFT multi-bin spectrogram classification
            </p>
          </div>
          <span className="font-mono text-xs font-bold px-3 py-1 bg-[#000000] text-[#ff4f2b] border border-[#3c3c3c] uppercase">
            6 PRE-VALIDATED CLIPS
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {PRELOADED_SAMPLES.map((sample) => {
            const isSelected = activeSample?.id === sample.id;
            return (
              <button
                key={sample.id}
                onClick={() => handleSampleSelect(sample)}
                className={`p-4 border text-left transition-all relative ${
                  isSelected
                    ? "bg-[#000000] border-[#ff4f2b] text-[#f5f5f5]"
                    : "bg-[#000000] border-[#3c3c3c] hover:border-[#f5f5f5]"
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="font-mono font-bold text-xs uppercase text-[#f5f5f5]">{sample.displayName}</span>
                  <Play className={`w-4 h-4 ${isSelected ? "text-[#ff4f2b]" : "text-[#737373]"}`} />
                </div>

                <p className="text-xs text-[#bfbfbf] mb-3 line-clamp-2">{sample.description}</p>

                <div className="flex items-center gap-2 text-[11px] text-[#737373] font-mono">
                  <span className="text-[#ff4f2b]">{sample.duration}</span>
                  <span>•</span>
                  <span className="uppercase">{sample.gender} ({sample.ageMonths}M)</span>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Manual File Upload Dropzone */}
      <div className="bg-[#1a1a1a] border border-[#3c3c3c] p-6">
        <label className="flex flex-col items-center justify-center p-8 border border-dashed border-[#3c3c3c] hover:border-[#ff4f2b] bg-[#000000] cursor-pointer transition group">
          <UploadCloud className="w-10 h-10 text-[#737373] group-hover:text-[#ff4f2b] transition mb-2" />
          <span className="font-mono font-bold text-sm text-[#f5f5f5] uppercase tracking-wider">Upload Custom Audio File</span>
          <span className="font-mono text-xs text-[#737373] mt-1">SUPPORTS WAV, MP3, M4A, CAF, 3GP (UP TO 25MB)</span>
          <input
            type="file"
            accept="audio/*,.wav,.m4a,.caf,.3gp,.mp3"
            onChange={handleFileUpload}
            className="hidden"
          />
        </label>

        {selectedFile && (
          <div className="mt-4 p-4 bg-[#000000] border border-[#3c3c3c] flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs text-[#f5f5f5] font-mono">
            <div className="flex items-center gap-3 truncate">
              <FileAudio className="w-5 h-5 text-[#ff4f2b] shrink-0" />
              <div className="truncate">
                <span className="font-bold text-[#f5f5f5] block truncate uppercase">{selectedFile.name}</span>
                <span className="text-[#737373]">{(selectedFile.size / 1024).toFixed(1)} KB • {selectedFile.type || "AUDIO/WAV"}</span>
              </div>
            </div>
            <audio
              controls
              src={URL.createObjectURL(selectedFile)}
              className="h-8 max-w-xs accent-[#ff4f2b]"
            />
          </div>
        )}
      </div>

      {/* Inference Results View */}
      <div className="bg-[#1a1a1a] border border-[#3c3c3c] p-6 space-y-5">
        <div className="flex items-center justify-between border-b border-[#3c3c3c] pb-3 font-mono">
          <h3 className="font-bold text-xs uppercase text-[#f5f5f5]">Classification Output Matrix</h3>
          <span className="text-[10px] text-[#737373]">MODEL: DEEPINFANT V2</span>
        </div>

        {isLoading ? (
          <div className="p-8 text-center space-y-3 font-mono">
            <div className="w-8 h-8 border-2 border-[#ff4f2b] border-t-transparent animate-spin mx-auto" />
            <p className="text-xs text-[#bfbfbf]">PROCESSING 16KHZ STFT SPECTROGRAM...</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Primary Suggestion Card */}
            <div className="bg-[#000000] border border-[#3c3c3c] p-6 space-y-4">
              <span className="font-mono text-xs font-bold text-[#737373] uppercase tracking-wider block">CLASSIFICATION_RESULT</span>

              <h2 className="font-sans font-light text-3xl sm:text-4xl uppercase text-[#f5f5f5] tracking-tight">
                {topMeta.displayName}
              </h2>

              <div className="flex items-center gap-3 font-mono text-xs">
                <span className="text-[#737373] uppercase">Confidence:</span>
                <span className="px-3 py-1 bg-[#ff4f2b] text-[#000000] font-bold">
                  {Math.round((analysisResult?.confidence || 0) * 100)}%
                </span>
              </div>

              <p className="text-xs sm:text-sm text-[#bfbfbf] leading-relaxed border-t border-[#3c3c3c] pt-3">
                {topMeta.description}
              </p>

              <div className="bg-[#1a1a1a] p-4 border border-[#3c3c3c] text-xs text-[#bfbfbf] space-y-1.5 font-mono">
                <span className="font-bold text-[#ff4f2b] block uppercase">RECOMMENDED CAREGIVER ACTION:</span>
                {topClass === "hungry" && <p>• Offer breast or bottle feeding. Check for root feeding reflexes.</p>}
                {topClass === "belly_pain" && <p>• Gently massage tummy clockwise, move legs in bicycle motion.</p>}
                {topClass === "burping" && <p>• Hold baby upright against shoulder and gently pat lower back.</p>}
                {topClass === "discomfort" && <p>• Check diaper status, adjust clothing layers or position.</p>}
                {topClass === "tired" && <p>• Reduce room noise/lights, swaddle gently, rock in quiet space.</p>}
                {topClass === "unknown" && <p>• Acoustic pattern ambiguous. Check hunger, diaper, and temperature manually.</p>}
              </div>
            </div>

            {/* Probability Bars */}
            <div className="bg-[#000000] border border-[#3c3c3c] p-6 space-y-4">
              <span className="font-mono text-xs font-bold text-[#737373] uppercase tracking-wider block">ALL 9 CRY CLASSES</span>

              <div className="space-y-3 font-mono text-xs">
                {analysisResult &&
                  Object.entries(analysisResult.probabilities)
                    .sort((a, b) => Number(b[1]) - Number(a[1]))
                    .map(([clsKey, prob]) => {
                      const meta = CRY_CLASS_METADATA[clsKey as CryClass] || CRY_CLASS_METADATA.unknown;
                      const percent = Math.round(Number(prob) * 100);

                      return (
                        <div key={clsKey} className="space-y-1">
                          <div className="flex justify-between font-medium text-[#bfbfbf]">
                            <span className="uppercase text-[#f5f5f5]">{meta.displayName}</span>
                            <span className="font-bold text-[#ff4f2b]">{percent}%</span>
                          </div>
                          <div className="h-2 bg-[#1a1a1a] border border-[#3c3c3c] overflow-hidden p-0">
                            <div
                              className="h-full bg-[#ff4f2b] transition-all duration-300"
                              style={{ width: `${percent}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

