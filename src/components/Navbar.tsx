import React from "react";
import { Activity, Mic, UploadCloud } from "lucide-react";

export type NavTab = "live" | "upload";

interface NavbarProps {
  activeTab: NavTab;
  setActiveTab: (tab: NavTab) => void;
  onOpenSafetyModal: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab
}) => {
  const tabs: { id: NavTab; label: string; icon: React.ReactNode }[] = [
    { id: "live", label: "LIVE ANALYSIS", icon: <Mic className="w-4 h-4" /> },
    { id: "upload", label: "AUDIO UPLOAD & SAMPLES", icon: <UploadCloud className="w-4 h-4" /> }
  ];

  return (
    <header className="sticky top-0 z-40 bg-[#000000] border-b border-[#3c3c3c] text-[#f5f5f5]">
      <div className="max-w-[1280px] mx-auto px-4 sm:px-6">
        <div className="flex items-center justify-between h-16 border-b border-[#3c3c3c]">
          {/* Brand Wordmark */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-[#ff4f2b] text-[#000000] flex items-center justify-center font-bold font-mono">
              <Activity className="w-5 h-5 text-[#000000]" />
            </div>
            <span className="font-mono font-bold text-xl tracking-wider uppercase text-[#f5f5f5]">
              DEEPINFANT<span className="text-[#ff4f2b]">_V2</span>
            </span>
          </div>

          <div className="flex items-center gap-2 font-mono text-xs text-[#737373]">
            <span className="hidden sm:inline uppercase">
              STATUS: <span className="text-[#ff4f2b] font-bold">IGNITED</span>
            </span>
          </div>
        </div>

        {/* Tab Strip */}
        <div className="flex gap-2 overflow-x-auto no-scrollbar py-3">
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-5 py-2.5 font-mono text-xs sm:text-sm tracking-wider uppercase transition-all border ${
                  isActive
                    ? "bg-[#ff4f2b] text-[#000000] border-[#ff4f2b] font-bold"
                    : "bg-[#1a1a1a] text-[#f5f5f5] border-[#3c3c3c] hover:border-[#f5f5f5] hover:bg-[#3c3c3c]"
                }`}
              >
                {tab.icon}
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </header>
  );
};


