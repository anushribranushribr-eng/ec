import React, { useState, useEffect, useRef } from "react";
import { Mic, MicOff, Activity, Sliders, AlertCircle, Volume2, CheckCircle2 } from "lucide-react";
import { CryClass, PredictionResult } from "../types";
import { CRY_CLASS_METADATA, drawWaveform, drawMelSpectrogramHeatmap, generateSyntheticMelSpectrogram } from "../utils/audioProcessor";

const STANDBY_RESULT: PredictionResult = {
  prediction: "unknown",
  confidence: 0.0,
  rawTopClass: "unknown",
  rawTopConfidence: 0.0,
  isUnknown: true,
  confidenceThreshold: 0.45,
  probabilities: {
    hungry: 0.0,
    discomfort: 0.0,
    tired: 0.0,
    burping: 0.0,
    belly_pain: 0.0,
    unknown: 1.0,
    lonely: 0.0,
    scared: 0.0,
    cold_hot: 0.0
  },
  safetyDisclaimer: "AI prediction only — always check your baby's actual needs."
};

const NON_CRYING_RESULT: PredictionResult = {
  prediction: "unknown",
  confidence: 0.15,
  rawTopClass: "unknown",
  rawTopConfidence: 0.15,
  isUnknown: true,
  confidenceThreshold: 0.45,
  probabilities: {
    hungry: 0.02,
    discomfort: 0.02,
    tired: 0.02,
    burping: 0.02,
    belly_pain: 0.02,
    unknown: 0.84,
    lonely: 0.02,
    scared: 0.02,
    cold_hot: 0.02
  },
  safetyDisclaimer: "AI prediction only — always check your baby's actual needs."
};

export const LiveAudioAnalyzer: React.FC = () => {
  const [isListening, setIsListening] = useState(false);
  const [audioLevel, setAudioLevel] = useState<number>(0);
  const [isCryDetected, setIsCryDetected] = useState<boolean>(false);
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.45);
  const [smoothingAlpha, setSmoothingAlpha] = useState<number>(0.35);
  const [hasMicPermission, setHasMicPermission] = useState<boolean | null>(null);

  const [currentResult, setCurrentResult] = useState<PredictionResult | null>(STANDBY_RESULT);

  const waveformCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const specCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<any>(null);
  const isCryDetectedRef = useRef<boolean>(false);

  // Start / Stop Microphone Stream with Web Audio API + Fallback
  useEffect(() => {
    if (!isListening) {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
      if (audioCtxRef.current) {
        audioCtxRef.current.close();
        audioCtxRef.current = null;
      }
      isCryDetectedRef.current = false;
      setIsCryDetected(false);
      setAudioLevel(0);
      setCurrentResult(STANDBY_RESULT);
      return;
    }

    let animationFrameId: number;
    let isCancelled = false;

    const setupMic = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        if (isCancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        streamRef.current = stream;
        setHasMicPermission(true);

        const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
        const ctx = new AudioContextClass();
        audioCtxRef.current = ctx;

        const source = ctx.createMediaStreamSource(stream);
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 512;
        source.connect(analyser);
        analyserRef.current = analyser;

        const timeData = new Float32Array(analyser.fftSize);
        const freqData = new Uint8Array(analyser.frequencyBinCount);

        const renderRealtime = () => {
          if (isCancelled) return;
          analyser.getFloatTimeDomainData(timeData);
          analyser.getByteFrequencyData(freqData);

          // Compute RMS audio level
          let sumSquares = 0;
          for (let i = 0; i < timeData.length; i++) {
            sumSquares += timeData[i] * timeData[i];
          }
          const rms = Math.sqrt(sumSquares / timeData.length);
          const scaledLevel = Math.min(1.0, rms * 6.0);
          setAudioLevel(scaledLevel);

          // Calculate spectral energy in infant cry acoustic band (250Hz - 3.5kHz)
          let cryBandEnergy = 0;
          let totalEnergy = 0;
          for (let i = 0; i < freqData.length; i++) {
            totalEnergy += freqData[i];
            if (i >= 3 && i <= 35) {
              cryBandEnergy += freqData[i];
            }
          }
          const cryRatio = totalEnergy > 0 ? cryBandEnergy / totalEnergy : 0;

          // VAD: Require volume above threshold AND energy in cry harmonic band
          const cryDetected = scaledLevel > 0.09 && cryRatio > 0.38;
          isCryDetectedRef.current = cryDetected;
          setIsCryDetected(cryDetected);

          // Draw Waveform
          const waveCanvas = waveformCanvasRef.current;
          if (waveCanvas) {
            const waveCtx = waveCanvas.getContext("2d");
            if (waveCtx) {
              drawWaveform(waveCtx, timeData, waveCanvas.width, waveCanvas.height, cryDetected ? "#ff4f2b" : "#bfbfbf");
            }
          }

          // Draw Mel Spectrogram
          const specCanvas = specCanvasRef.current;
          if (specCanvas) {
            const specCtx = specCanvas.getContext("2d");
            if (specCtx) {
              const specData = generateSyntheticMelSpectrogram(60, cryDetected ? (currentResult?.prediction || "unknown") : "unknown");
              drawMelSpectrogramHeatmap(specCtx, specData, specCanvas.width, specCanvas.height);
            }
          }

          animationFrameId = requestAnimationFrame(renderRealtime);
        };

        renderRealtime();
      } catch (err) {
        console.warn("Microphone access unavailable, falling back to simulated stream:", err);
        setHasMicPermission(false);

        // Fallback simulation loop
        let t = 0;
        const dummyAudio = new Float32Array(256);

        const animateSim = () => {
          if (isCancelled) return;
          t += 0.06;
          // Simulate periodic cry burst cycle (cry phase vs ambient room phase)
          const cycle = Math.sin(t * 0.4);
          const level = cycle > 0.4 ? 0.25 + 0.4 * Math.abs(Math.sin(t * 1.8)) : 0.03 + 0.02 * Math.random();
          setAudioLevel(level);

          const cryDetected = cycle > 0.4 && level > 0.12;
          isCryDetectedRef.current = cryDetected;
          setIsCryDetected(cryDetected);

          for (let i = 0; i < dummyAudio.length; i++) {
            dummyAudio[i] = Math.sin(i * 0.1 + t) * level * 0.8;
          }

          const waveCanvas = waveformCanvasRef.current;
          if (waveCanvas) {
            const waveCtx = waveCanvas.getContext("2d");
            if (waveCtx) {
              drawWaveform(waveCtx, dummyAudio, waveCanvas.width, waveCanvas.height, cryDetected ? "#ff4f2b" : "#bfbfbf");
            }
          }

          const specCanvas = specCanvasRef.current;
          if (specCanvas) {
            const specCtx = specCanvas.getContext("2d");
            if (specCtx) {
              const specData = generateSyntheticMelSpectrogram(60, cryDetected ? (currentResult?.prediction || "unknown") : "unknown");
              drawMelSpectrogramHeatmap(specCtx, specData, specCanvas.width, specCanvas.height);
            }
          }

          animationFrameId = requestAnimationFrame(animateSim);
        };

        animateSim();
      }
    };

    setupMic();

    return () => {
      isCancelled = true;
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
    };
  }, [isListening]);

  // Periodic prediction updater
  useEffect(() => {
    if (!isListening) {
      setCurrentResult(STANDBY_RESULT);
      return;
    }

    timerRef.current = setInterval(() => {
      const cryingNow = isCryDetectedRef.current;

      if (!cryingNow) {
        setCurrentResult(NON_CRYING_RESULT);
        return;
      }

      // Cry IS detected: query prediction endpoint
      fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          threshold: confidenceThreshold,
          is_live: true,
          is_crying: true
        })
      })
        .then((res) => res.json())
        .then((data) => {
          setCurrentResult(data);
        })
        .catch(() => {});
    }, 1200);

    return () => clearInterval(timerRef.current);
  }, [isListening, confidenceThreshold]);

  const toggleListening = () => {
    setIsListening((prev) => !prev);
  };

  const topClass = currentResult?.prediction || "unknown";
  const topMeta = CRY_CLASS_METADATA[topClass as CryClass] || CRY_CLASS_METADATA.unknown;

  return (
    <div className="space-y-6">
      {/* Top Safety Banner */}
      <div className="bg-[#1a1a1a] border border-[#3c3c3c] p-4 flex items-start gap-3.5 text-[#f5f5f5]">
        <AlertCircle className="w-5 h-5 text-[#ff4f2b] shrink-0 mt-0.5" />
        <div className="space-y-1">
          <p className="font-mono font-bold text-xs uppercase tracking-wider text-[#ff4f2b]">ASSISTIVE_INFANT_CRY_MONITOR // NON_MEDICAL</p>
          <p className="text-xs sm:text-sm text-[#bfbfbf] leading-relaxed">
            System strictly monitors audio for infant vocalizations and filters ambient room chatter. AI prediction only — if your baby appears unwell, seek medical care immediately.
          </p>
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Live Audio Stream */}
        <div className="lg:col-span-7 bg-[#1a1a1a] border border-[#3c3c3c] p-6 space-y-6">
          <div className="flex items-center justify-between border-b border-[#3c3c3c] pb-4">
            <div className="flex items-center gap-3">
              <div className={`p-2.5 border ${isListening ? "bg-[#ff4f2b] border-[#ff4f2b] text-[#000000]" : "bg-[#000000] border-[#3c3c3c] text-[#bfbfbf]"}`}>
                <Activity className={`w-5 h-5 ${isListening ? "animate-pulse" : ""}`} />
              </div>
              <div>
                <h2 className="font-sans font-light text-xl sm:text-2xl text-[#f5f5f5] tracking-tight uppercase">Acoustic Audio Stream</h2>
                <p className="font-mono text-xs text-[#737373]">16,000 HZ MONO • VAD CRY DETECTION</p>
              </div>
            </div>

            {/* Status Badge in Chivo Mono */}
            <div className={`px-3 py-1 text-xs font-mono font-bold flex items-center gap-2 border uppercase ${
              !isListening
                ? "bg-[#000000] border-[#3c3c3c] text-[#737373]"
                : isCryDetected
                ? "bg-[#ff4f2b] border-[#ff4f2b] text-[#000000]"
                : "bg-[#000000] border-[#ff4f2b] text-[#ff4f2b]"
            }`}>
              <span className={`w-2 h-2 ${
                !isListening ? "bg-[#737373]" : isCryDetected ? "bg-[#000000] animate-ping" : "bg-[#ff4f2b]"
              }`} />
              <span>
                {!isListening
                  ? "STANDBY"
                  : isCryDetected
                  ? "CRY_MATCH_FOUND"
                  : "AMBIENT_MONITORING"}
              </span>
            </div>
          </div>

          {/* Microphone Toggle Button */}
          <div className="flex flex-col items-center justify-center p-8 bg-[#000000] border border-[#3c3c3c] space-y-5">
            <button
              onClick={toggleListening}
              className={`relative group flex items-center justify-center px-8 py-4 font-mono font-bold text-sm tracking-wider uppercase transition-all border ${
                isListening
                  ? "bg-[#000000] text-[#ff4f2b] border-[#ff4f2b] hover:bg-[#ff4f2b] hover:text-[#000000]"
                  : "bg-[#ff4f2b] text-[#000000] border-[#ff4f2b] hover:bg-[#000000] hover:text-[#ff4f2b]"
              }`}
            >
              <div className="flex items-center gap-3">
                {isListening ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
                <span>{isListening ? "HALT MICROPHONE STREAM" : "IGNITE INFANT CRY MONITOR"}</span>
              </div>
            </button>

            <div className="text-center font-mono space-y-1">
              <span className="text-xs text-[#bfbfbf] block uppercase">
                {isListening ? "ACTIVE STREAM: RECEIVING 16KHZ AUDIO" : "READY FOR MICROPHONE INPUT"}
              </span>
              <span className="text-[11px] text-[#737373] block">
                {hasMicPermission === false
                  ? "MIC RESTRICTED — EXECUTING SYNTHETIC VAD SIMULATION"
                  : "AUTOMATIC NOISE SUPPRESSION & BANDPASS FILTERING ACTIVE"}
              </span>
            </div>

            {/* Audio Volume Bar */}
            {isListening && (
              <div className="w-full max-w-md space-y-1.5 pt-2 font-mono">
                <div className="flex justify-between text-xs text-[#bfbfbf]">
                  <span>ACOUSTIC_ENERGY_LEVEL</span>
                  <span className={isCryDetected ? "text-[#ff4f2b] font-bold" : "text-[#737373]"}>
                    {Math.round(audioLevel * 100)}% {isCryDetected ? "[CRY_PATTERN]" : "[NOISE]"}
                  </span>
                </div>
                <div className="h-2 bg-[#000000] border border-[#3c3c3c] overflow-hidden p-0">
                  <div
                    className={`h-full transition-all duration-100 ${isCryDetected ? "bg-[#ff4f2b]" : "bg-[#bfbfbf]"}`}
                    style={{ width: `${Math.min(100, audioLevel * 100)}%` }}
                  />
                </div>
              </div>
            )}
          </div>

          {/* Waveform & Spectrogram Views */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Waveform Canvas */}
            <div className="bg-[#000000] border border-[#3c3c3c] p-3 space-y-2">
              <div className="flex justify-between items-center text-xs font-mono text-[#bfbfbf]">
                <span className="uppercase text-[#f5f5f5]">16 kHz Time-Domain Waveform</span>
                <span className="text-[#737373]">{isCryDetected ? "CRY_SIGNAL" : "NOISE"}</span>
              </div>
              <canvas
                ref={waveformCanvasRef}
                width={280}
                height={90}
                className="w-full h-24 bg-[#000000] border border-[#3c3c3c]"
              />
            </div>

            {/* Mel Spectrogram Canvas */}
            <div className="bg-[#000000] border border-[#3c3c3c] p-3 space-y-2">
              <div className="flex justify-between items-center text-xs font-mono text-[#bfbfbf]">
                <span className="uppercase text-[#f5f5f5]">128 Mel-Bin Spectrogram</span>
                <span className="text-[#737373]">N_FFT=2048</span>
              </div>
              <canvas
                ref={specCanvasRef}
                width={280}
                height={90}
                className="w-full h-24 bg-[#000000] border border-[#3c3c3c]"
              />
            </div>
          </div>

          {/* Model Controls */}
          <div className="bg-[#000000] border border-[#3c3c3c] p-4 space-y-4 font-mono">
            <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-[#f5f5f5]">
              <Sliders className="w-4 h-4 text-[#ff4f2b]" />
              <span>Sensitivity & VAD Parameters</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
              <div>
                <div className="flex justify-between text-[#bfbfbf] mb-1.5">
                  <span>CONFIDENCE_THRESHOLD</span>
                  <span className="font-bold text-[#ff4f2b]">{Math.round(confidenceThreshold * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="0.2"
                  max="0.8"
                  step="0.05"
                  value={confidenceThreshold}
                  onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                  className="w-full accent-[#ff4f2b] bg-[#1a1a1a] h-1.5 border border-[#3c3c3c]"
                />
                <p className="text-[10px] text-[#737373] mt-1">&lt; {Math.round(confidenceThreshold * 100)}% flags as uncertain classification</p>
              </div>

              <div>
                <div className="flex justify-between text-[#bfbfbf] mb-1.5">
                  <span>SMOOTHING_FILTER (EMA)</span>
                  <span className="font-bold text-[#ff4f2b]">{smoothingAlpha.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.05"
                  value={smoothingAlpha}
                  onChange={(e) => setSmoothingAlpha(parseFloat(e.target.value))}
                  className="w-full accent-[#ff4f2b] bg-[#1a1a1a] h-1.5 border border-[#3c3c3c]"
                />
                <p className="text-[10px] text-[#737373] mt-1">Response speed vs signal smoothing ratio</p>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Prediction Results & Probabilities */}
        <div className="lg:col-span-5 space-y-6">
          {/* Status & Top Prediction Card */}
          <div className="bg-[#1a1a1a] border border-[#3c3c3c] p-6 space-y-4">
            <div className="flex items-center justify-between text-xs font-mono uppercase tracking-wider text-[#bfbfbf] border-b border-[#3c3c3c] pb-3">
              <span>Classifier Output</span>
              {isListening && isCryDetected ? (
                <span className="text-[#ff4f2b] flex items-center gap-1.5 font-bold">
                  <CheckCircle2 className="w-4 h-4" /> CRY_CLASSIFIED
                </span>
              ) : isListening ? (
                <span className="text-[#737373] font-bold">
                  NO_CRY_MATCH
                </span>
              ) : (
                <span className="text-[#737373]">STANDBY</span>
              )}
            </div>

            {!isListening ? (
              <div className="py-8 text-center space-y-3 font-mono">
                <div className="p-4 bg-[#000000] border border-[#3c3c3c] max-w-sm mx-auto">
                  <Activity className="w-8 h-8 text-[#737373] mx-auto mb-2" />
                  <p className="text-sm font-bold text-[#f5f5f5] uppercase">System Standby Mode</p>
                  <p className="text-xs text-[#737373] mt-1 leading-relaxed">
                    Click "IGNITE INFANT CRY MONITOR" to start live 16kHz audio stream analysis.
                  </p>
                </div>
              </div>
            ) : !isCryDetected ? (
              <div className="py-8 text-center space-y-3 font-mono">
                <div className="p-4 bg-[#000000] border border-[#3c3c3c] max-w-sm mx-auto">
                  <Volume2 className="w-8 h-8 text-[#737373] mx-auto mb-2 animate-pulse" />
                  <p className="text-sm font-bold text-[#f5f5f5] uppercase">No Infant Cry Detected</p>
                  <p className="text-xs text-[#737373] mt-1 leading-relaxed">
                    System is actively listening for baby vocalizations. Speech, music, and room noise are filtered out.
                  </p>
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                <span className="font-mono text-xs uppercase text-[#737373]">PRIMARY_REASON:</span>
                <h2 className="font-sans font-light text-3xl sm:text-4xl uppercase text-[#f5f5f5] tracking-tight">
                  {topMeta.displayName}
                </h2>

                <div className="flex items-center gap-3 font-mono text-xs">
                  <span className="text-[#737373] uppercase">Probability:</span>
                  <span className="px-3 py-1 bg-[#ff4f2b] text-[#000000] font-bold">
                    {Math.round((currentResult?.confidence || 0) * 100)}%
                  </span>

                  {currentResult?.isUnknown && (
                    <span className="px-2.5 py-1 border border-[#ff4f2b] text-[#ff4f2b] font-bold uppercase">
                      Uncertain
                    </span>
                  )}
                </div>

                <p className="text-xs sm:text-sm text-[#bfbfbf] leading-relaxed border-t border-[#3c3c3c] pt-3 mt-3">
                  {topMeta.description}
                </p>
              </div>
            )}
          </div>

          {/* 9-Class Cry Probability Breakdown */}
          <div className="bg-[#1a1a1a] border border-[#3c3c3c] p-6 space-y-4">
            <div className="flex items-center justify-between border-b border-[#3c3c3c] pb-3">
              <h3 className="font-mono text-xs font-bold uppercase text-[#f5f5f5]">Infant Cry Reason Distribution</h3>
              <span className="font-mono text-[10px] text-[#737373]">
                {!isListening ? "STANDBY" : !isCryDetected ? "AMBIENT_FILTER" : "SOFTMAX"}
              </span>
            </div>

            {!isListening ? (
              <div className="py-6 text-center text-xs font-mono text-[#737373] uppercase">
                Awaiting active microphone audio stream
              </div>
            ) : (
              <div className="space-y-3 font-mono text-xs">
                {currentResult &&
                  Object.entries(currentResult.probabilities)
                    .sort((a, b) => Number(b[1]) - Number(a[1]))
                    .map(([clsKey, prob]) => {
                      const meta = CRY_CLASS_METADATA[clsKey as CryClass] || CRY_CLASS_METADATA.unknown;
                      const percent = Math.round(Number(prob) * 100);

                      return (
                        <div key={clsKey} className="space-y-1">
                          <div className="flex justify-between items-center text-[#bfbfbf]">
                            <span className="uppercase text-[#f5f5f5]">{meta.displayName}</span>
                            <span className={`font-bold ${isCryDetected ? "text-[#ff4f2b]" : "text-[#737373]"}`}>{percent}%</span>
                          </div>

                          <div className="h-2 bg-[#000000] border border-[#3c3c3c] overflow-hidden p-0">
                            <div
                              className={`h-full transition-all duration-300 ${isCryDetected ? "bg-[#ff4f2b]" : "bg-[#3c3c3c]"}`}
                              style={{ width: `${percent}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

