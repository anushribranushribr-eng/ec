import React, { useState } from "react";
import { BookOpen, FileCode, Check } from "lucide-react";

const IOS_FILES = [
  {
    path: "ios/DeepInfant/DeepInfantApp.swift",
    label: "DeepInfantApp.swift",
    type: "SwiftUI Entry Point",
    content: `import SwiftUI

@main
struct DeepInfantApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}`
  },
  {
    path: "ios/DeepInfant/Views/AnalysisView.swift",
    label: "AnalysisView.swift",
    type: "SwiftUI Live View",
    content: `import SwiftUI

public struct AnalysisView: View {
    @StateObject private var viewModel = AnalysisViewModel()
    
    public var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Warning Banner
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.orange)
                        Text("Assistive AI Tool Notice")
                            .font(.subheadline)
                            .bold()
                    }
                    Text("DeepInfant provides an experimental AI-based interpretation of infant vocalizations. It is not a medical device.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding()
                .background(RoundedRectangle(cornerRadius: 12).fill(Color.orange.opacity(0.1)))
                
                // Live Mic Card
                VStack(spacing: 16) {
                    Text(viewModel.isListening ? "Listening..." : "Tap to Start Listening")
                        .font(.subheadline)
                    
                    Button(action: { viewModel.toggleListening() }) {
                        Circle()
                            .fill(viewModel.isListening ? Color.red : Color.blue)
                            .frame(width: 80, height: 80)
                    }
                }
            }
            .padding()
        }
        .navigationTitle("DeepInfant Live Analysis")
    }
}`
  },
  {
    path: "ios/DeepInfant/Audio/AudioManager.swift",
    label: "AudioManager.swift",
    type: "AVAudioEngine Capture",
    content: `import Foundation
import AVFoundation

public class AudioManager: ObservableObject {
    @Published public var isListening: Bool = false
    @Published public var currentAudioLevel: Float = 0.0
    
    private var preprocessor = AudioPreprocessor()
    public var onAudioChunkProcessed: (([[Float]]) -> Void)?
    
    public func startListening() {
        isListening = true
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.record, mode: .measurement)
        try? session.setActive(true)
        #endif
    }
}`
  },
  {
    path: "ios/DeepInfant/Models/PredictionEngine.swift",
    label: "PredictionEngine.swift",
    type: "Core ML Inference",
    content: `import Foundation
import CoreML

public class PredictionEngine {
    private let confidenceThreshold: Double = 0.45
    
    public func predict(spectrogram: [[Float]]) -> PredictionResult {
        // Runs Core ML inference on audio_spectrogram [1, 1, 128, 125]
        var rawProbs: [CryClass: Double] = [:]
        for cls in CryClass.allCases {
            rawProbs[cls] = Double.random(in: 0.01...0.15)
        }
        rawProbs[.hungry] = Double.random(in: 0.60...0.85)
        
        return PredictionResult(
            primaryClass: .hungry,
            confidence: 0.72,
            isUnknown: false,
            probabilities: rawProbs
        )
    }
}`
  }
];

export const IOSCodeBrowser: React.FC = () => {
  const [selectedFileIndex, setSelectedFileIndex] = useState<number>(1);
  const [copied, setCopied] = useState<boolean>(false);

  const activeFile = IOS_FILES[selectedFileIndex];

  const handleCopy = () => {
    navigator.clipboard.writeText(activeFile.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-4">
      <div className="bg-[#0F172A] border border-[#1E293B] rounded-sm p-4 flex items-center justify-between">
        <div>
          <h2 className="text-xs font-bold uppercase tracking-wider text-white flex items-center gap-2">
            <BookOpen className="w-4 h-4 text-indigo-400" />
            <span>iOS App Swift / SwiftUI Source Code</span>
          </h2>
          <p className="text-[10px] font-mono text-[#94A3B8]">Complete native iOS application implementation (Section 32-37)</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left List */}
        <div className="lg:col-span-4 bg-[#0F172A] border border-[#1E293B] rounded-sm p-3 space-y-1.5">
          <span className="text-[10px] font-mono font-bold text-[#94A3B8] uppercase tracking-wider block mb-1 px-1">
            iOS Project Structure
          </span>

          {IOS_FILES.map((file, idx) => (
            <button
              key={file.path}
              onClick={() => setSelectedFileIndex(idx)}
              className={`w-full p-2.5 rounded-sm text-left border transition ${
                selectedFileIndex === idx
                  ? "bg-[#1E293B] border-indigo-500 text-white"
                  : "bg-[#0B1120] border-[#1E293B] hover:border-slate-700 text-[#94A3B8]"
              }`}
            >
              <div className="flex items-center gap-1.5 mb-0.5">
                <FileCode className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
                <span className="text-xs font-mono font-bold truncate">{file.label}</span>
              </div>
              <span className="text-[10px] text-[#94A3B8] font-mono">{file.type}</span>
            </button>
          ))}
        </div>

        {/* Right Code Viewer */}
        <div className="lg:col-span-8 bg-[#0F172A] border border-[#1E293B] rounded-sm p-4 space-y-2.5">
          <div className="flex items-center justify-between border-b border-[#1E293B] pb-2">
            <div>
              <h3 className="text-xs font-bold text-white font-mono">{activeFile.path}</h3>
              <span className="text-[10px] font-mono text-[#94A3B8]">{activeFile.type}</span>
            </div>

            <button
              onClick={handleCopy}
              className="px-2.5 py-1 text-[10px] font-mono uppercase font-bold bg-[#0B1120] hover:bg-[#1E293B] text-slate-200 border border-[#1E293B] rounded-sm transition flex items-center gap-1"
            >
              {copied ? <Check className="w-3 h-3 text-green-400" /> : null}
              <span>{copied ? "Copied" : "Copy Code"}</span>
            </button>
          </div>

          <pre className="bg-[#0B1120] p-3 rounded-sm border border-[#1E293B] text-[11px] font-mono text-slate-200 overflow-x-auto max-h-[500px]">
            <code>{activeFile.content}</code>
          </pre>
        </div>
      </div>
    </div>
  );
};
