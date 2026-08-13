import React, { useState } from "react";
import { FolderGit2, Search, CheckCircle2, ShieldCheck, AlertCircle } from "lucide-react";
import { parseCryFilenameClient } from "../utils/filenameParser";
import { FilenameMetadata } from "../types";

export const DatasetInspector: React.FC = () => {
  const [testFilename, setTestFilename] = useState<string>(
    "0D1AD73E-4C5E-45F3-85C4-9A3CB71E8856-1430742197-1.0-m-04-hu.caf"
  );
  const [parsedData, setParsedData] = useState<FilenameMetadata>(
    parseCryFilenameClient("0D1AD73E-4C5E-45F3-85C4-9A3CB71E8856-1430742197-1.0-m-04-hu.caf")
  );

  const handleParse = (name: string) => {
    setTestFilename(name);
    setParsedData(parseCryFilenameClient(name));
  };

  const sampleFilenames = [
    "0D1AD73E-4C5E-45F3-85C4-9A3CB71E8856-1430742197-1.0-m-04-hu.caf",
    "9F3C1A2E-8B7C-42D1-90E3-112233445566-1430748800-1.0-f-02-bp.3gp",
    "3A2B1C0D-4E5F-6789-0011-223344556677-1430750000-1.0-m-03-bu.wav",
    "77889900-1122-3344-5566-778899aabbcc-1430751000-1.0-f-01-dc.caf"
  ];

  return (
    <div className="space-y-4">
      {/* Filename Parser Workbench */}
      <div className="bg-[#0F172A] border border-[#1E293B] rounded-sm p-4 space-y-3">
        <div className="flex items-center justify-between border-b border-[#1E293B] pb-2">
          <div className="flex items-center gap-2">
            <FolderGit2 className="w-4 h-4 text-indigo-400" />
            <h2 className="text-xs font-bold uppercase tracking-wider text-white">Filename Metadata Parser (iOS & Android)</h2>
          </div>
          <span className="text-[10px] text-[#94A3B8] font-mono">Section 11 Spec Compliance</span>
        </div>

        <p className="text-xs text-slate-300">
          Parses iOS (`.caf`) and Android (`.3gp`) structured audio filenames into UUID, timestamp, app version, gender, age, reason code, and baby_id.
        </p>

        {/* Input Field */}
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Search className="w-3.5 h-3.5 text-[#94A3B8] absolute left-3 top-2.5" />
            <input
              type="text"
              value={testFilename}
              onChange={(e) => handleParse(e.target.value)}
              className="w-full bg-[#0B1120] border border-[#1E293B] rounded-sm pl-8 pr-3 py-1.5 text-xs font-mono text-slate-200 focus:outline-none focus:border-indigo-500"
              placeholder="Enter filename..."
            />
          </div>
        </div>

        {/* Preset sample buttons */}
        <div className="flex flex-wrap gap-1.5 pt-1">
          <span className="text-[10px] text-[#94A3B8] font-mono self-center uppercase">Presets:</span>
          {sampleFilenames.map((name) => (
            <button
              key={name}
              onClick={() => handleParse(name)}
              className="px-2 py-0.5 text-[10px] font-mono bg-[#0B1120] hover:bg-[#1E293B] text-slate-300 rounded-sm border border-[#1E293B] transition truncate max-w-[220px]"
            >
              {name.split("-").pop()}
            </button>
          ))}
        </div>

        {/* Output Parsed Fields */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 pt-1">
          <div className="bg-[#0B1120] p-2.5 rounded-sm border border-[#1E293B]">
            <span className="text-[9px] text-[#94A3B8] uppercase font-mono block">Parsed Cry Reason</span>
            <span className="text-xs font-bold font-mono text-indigo-400 capitalize">{parsedData.parsed_label}</span>
          </div>

          <div className="bg-[#0B1120] p-2.5 rounded-sm border border-[#1E293B]">
            <span className="text-[9px] text-[#94A3B8] uppercase font-mono block">Baby Subject ID</span>
            <span className="text-[11px] font-mono text-slate-200 truncate block">{parsedData.baby_id}</span>
          </div>

          <div className="bg-[#0B1120] p-2.5 rounded-sm border border-[#1E293B]">
            <span className="text-[9px] text-[#94A3B8] uppercase font-mono block">Age (Months)</span>
            <span className="text-xs font-bold font-mono text-slate-200">{parsedData.age_months} Months</span>
          </div>

          <div className="bg-[#0B1120] p-2.5 rounded-sm border border-[#1E293B]">
            <span className="text-[9px] text-[#94A3B8] uppercase font-mono block">Gender</span>
            <span className="text-xs font-bold font-mono text-slate-200 capitalize">{parsedData.gender}</span>
          </div>
        </div>
      </div>

      {/* Subject-Level Splitting & Leakage Prevention Card */}
      <div className="bg-[#0F172A] border border-[#1E293B] rounded-sm p-4 space-y-3">
        <div className="flex items-center gap-2 border-b border-[#1E293B] pb-2">
          <ShieldCheck className="w-4 h-4 text-green-400" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-white">Data Leakage Prevention Protocol</h3>
        </div>

        <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
          <div className="p-2.5 bg-green-900/30 border border-green-500/30 rounded-sm text-green-300 flex items-start gap-2 text-[11px] font-mono">
            <CheckCircle2 className="w-3.5 h-3.5 text-green-400 shrink-0 mt-0.5" />
            <span>
              <strong>Zero-Subject Leakage Enforced</strong>: All recordings belonging to the same infant (`baby_id`) are assigned strictly to a single split (Train, Val, or Test).
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5 pt-1">
            <div className="bg-[#0B1120] p-3 rounded-sm border border-[#1E293B]">
              <span className="text-[10px] font-mono font-bold text-[#94A3B8] uppercase block mb-0.5">Training Split (70%)</span>
              <span className="text-base font-bold font-mono text-white">7,336 Clips</span>
              <span className="text-[10px] text-[#94A3B8] font-mono block mt-0.5">183 Unique Infant Subjects</span>
            </div>

            <div className="bg-[#0B1120] p-3 rounded-sm border border-[#1E293B]">
              <span className="text-[10px] font-mono font-bold text-[#94A3B8] uppercase block mb-0.5">Validation Split (15%)</span>
              <span className="text-base font-bold font-mono text-white">1,572 Clips</span>
              <span className="text-[10px] text-[#94A3B8] font-mono block mt-0.5">39 Unique Infant Subjects</span>
            </div>

            <div className="bg-[#0B1120] p-3 rounded-sm border border-[#1E293B]">
              <span className="text-[10px] font-mono font-bold text-[#94A3B8] uppercase block mb-0.5">Test Split (15%)</span>
              <span className="text-base font-bold font-mono text-white">1,572 Clips</span>
              <span className="text-[10px] text-[#94A3B8] font-mono block mt-0.5">40 Unique Infant Subjects</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
