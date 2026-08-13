import React, { useState } from "react";
import { Navbar, NavTab } from "./components/Navbar";
import { SafetyDisclaimerModal } from "./components/SafetyDisclaimerModal";
import { LiveAudioAnalyzer } from "./components/LiveAudioAnalyzer";
import { AudioFileUpload } from "./components/AudioFileUpload";
import { HeartPulse } from "lucide-react";

export function App() {
  const [activeTab, setActiveTab] = useState<NavTab>("live");
  const [isSafetyModalOpen, setIsSafetyModalOpen] = useState<boolean>(false);

  return (
    <div className="min-h-screen bg-[#000000] text-[#f5f5f5] flex flex-col font-sans selection:bg-[#ff4f2b] selection:text-[#000000]">
      {/* Navigation Header */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onOpenSafetyModal={() => setIsSafetyModalOpen(true)}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-[1280px] w-full mx-auto px-4 sm:px-6 py-8">
        {activeTab === "live" && <LiveAudioAnalyzer />}
        {activeTab === "upload" && <AudioFileUpload />}
      </main>

      {/* Safety Modal Popup */}
      <SafetyDisclaimerModal
        isOpen={isSafetyModalOpen}
        onClose={() => setIsSafetyModalOpen(false)}
      />

      {/* Footer */}
      <footer className="bg-[#000000] border-t border-[#3c3c3c] text-[#737373] py-8 text-xs font-mono">
        <div className="max-w-[1280px] mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3 text-[#f5f5f5]">
            <div className="w-2 h-2 bg-[#ff4f2b]"></div>
            <span className="font-bold text-sm tracking-wider uppercase">DEEPINFANT_V2</span>
            <span className="text-[#3c3c3c]">|</span>
            <span className="text-[#737373]">VOICE_ACTIVITY_DETECTOR</span>
          </div>

          <div className="flex items-center gap-4 text-[#bfbfbf]">
            <span>INFRASTRUCTURE_STATE: ONLINE</span>
            <span className="text-[#3c3c3c]">|</span>
            <span className="text-[#ff4f2b]">16KHZ_MONO</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;


