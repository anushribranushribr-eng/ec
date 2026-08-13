import React, { useState, useEffect } from "react";
import { BarChart3, CheckCircle2, ShieldCheck, Database, Award } from "lucide-react";
import { BenchmarkMetrics, CryClass, PerClassMetric } from "../types";
import { CRY_CLASS_METADATA } from "../utils/audioProcessor";

export const EvaluationDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<BenchmarkMetrics | null>(null);

  useEffect(() => {
    fetch("/api/benchmark")
      .then((res) => res.json())
      .then((data) => setMetrics(data))
      .catch((err) => console.error(err));
  }, []);

  if (!metrics) {
    return (
      <div className="p-8 text-center text-[#94A3B8] font-mono text-xs">
        <div className="w-6 h-6 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto mb-2" />
        <span>Loading Benchmark Evaluation Metrics...</span>
      </div>
    );
  }

  const classes: CryClass[] = [
    "belly_pain",
    "burping",
    "cold_hot",
    "discomfort",
    "hungry",
    "lonely",
    "scared",
    "tired",
    "unknown"
  ];

  return (
    <div className="space-y-4">
      {/* Top Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="bg-[#0F172A] border border-[#1E293B] p-3.5 rounded-sm">
          <div className="flex justify-between items-center text-[10px] font-mono font-bold uppercase tracking-wider text-[#94A3B8] mb-1">
            <span>Overall Accuracy</span>
            <Award className="w-3.5 h-3.5 text-green-400" />
          </div>
          <div className="text-xl font-bold font-mono text-white">{(metrics.accuracy * 100).toFixed(2)}%</div>
          <p className="text-[10px] text-green-400 mt-0.5 font-mono">Test Set (1,572 samples)</p>
        </div>

        <div className="bg-[#0F172A] border border-[#1E293B] p-3.5 rounded-sm">
          <div className="flex justify-between items-center text-[10px] font-mono font-bold uppercase tracking-wider text-[#94A3B8] mb-1">
            <span>Macro F1-Score</span>
            <BarChart3 className="w-3.5 h-3.5 text-indigo-400" />
          </div>
          <div className="text-xl font-bold font-mono text-white">{metrics.macro_f1.toFixed(4)}</div>
          <p className="text-[10px] text-[#94A3B8] mt-0.5 font-mono">Unweighted Class Average</p>
        </div>

        <div className="bg-[#0F172A] border border-[#1E293B] p-3.5 rounded-sm">
          <div className="flex justify-between items-center text-[10px] font-mono font-bold uppercase tracking-wider text-[#94A3B8] mb-1">
            <span>Weighted F1-Score</span>
            <BarChart3 className="w-3.5 h-3.5 text-indigo-400" />
          </div>
          <div className="text-xl font-bold font-mono text-white">{metrics.weighted_f1.toFixed(4)}</div>
          <p className="text-[10px] text-[#94A3B8] mt-0.5 font-mono">Class Imbalance Adjusted</p>
        </div>

        <div className="bg-[#0F172A] border border-[#1E293B] p-3.5 rounded-sm">
          <div className="flex justify-between items-center text-[10px] font-mono font-bold uppercase tracking-wider text-[#94A3B8] mb-1">
            <span>Subject Leakage</span>
            <ShieldCheck className="w-3.5 h-3.5 text-green-400" />
          </div>
          <div className="text-xl font-bold font-mono text-green-400">{metrics.subject_leakage_percentage.toFixed(2)}%</div>
          <p className="text-[10px] text-green-400 mt-0.5 font-mono">Strict Subject Split Verified</p>
        </div>
      </div>

      {/* Per-Class Metrics Table */}
      <div className="bg-[#0F172A] border border-[#1E293B] rounded-sm p-4 space-y-3">
        <div className="flex items-center justify-between border-b border-[#1E293B] pb-2">
          <h3 className="text-xs font-bold uppercase tracking-wider text-white">Per-Class Performance Breakdown</h3>
          <span className="text-[10px] font-mono text-[#94A3B8]">9 Predefined Cry Classes</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left text-slate-300">
            <thead className="bg-[#0B1120] text-[#94A3B8] uppercase font-mono text-[10px] border-b border-[#1E293B]">
              <tr>
                <th className="p-2.5">Cry Class</th>
                <th className="p-2.5">Precision</th>
                <th className="p-2.5">Recall</th>
                <th className="p-2.5">F1-Score</th>
                <th className="p-2.5">Support</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1E293B]">
              {Object.entries(metrics.per_class).map(([clsKey, rawVal]) => {
                const val = rawVal as PerClassMetric;
                const meta = CRY_CLASS_METADATA[clsKey as CryClass] || CRY_CLASS_METADATA.unknown;
                return (
                  <tr key={clsKey} className="hover:bg-[#1E293B]/40">
                    <td className="p-2.5 font-semibold text-white flex items-center gap-2">
                      <span className="w-2 h-2 rounded-sm" style={{ backgroundColor: meta.color }} />
                      <span>{meta.displayName}</span>
                    </td>
                    <td className="p-2.5 font-mono text-[11px]">{val.precision.toFixed(4)}</td>
                    <td className="p-2.5 font-mono text-[11px]">{val.recall.toFixed(4)}</td>
                    <td className="p-2.5 font-mono text-[11px] font-bold text-indigo-400">{val.f1_score.toFixed(4)}</td>
                    <td className="p-2.5 font-mono text-[11px] text-[#94A3B8]">{val.support}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Confusion Matrix Heatmap View */}
      <div className="bg-[#0F172A] border border-[#1E293B] rounded-sm p-4 space-y-3">
        <div className="flex items-center justify-between border-b border-[#1E293B] pb-2">
          <h3 className="text-xs font-bold uppercase tracking-wider text-white">Confusion Matrix Heatmap</h3>
          <span className="text-[10px] font-mono text-[#94A3B8]">Rows: Actual • Columns: Predicted</span>
        </div>

        <div className="overflow-x-auto p-1">
          <div className="min-w-[580px] grid grid-cols-10 gap-1 text-[10px] font-mono">
            {/* Corner header */}
            <div className="p-1.5 text-slate-500 font-bold">Act \ Pred</div>
            {classes.map((c) => (
              <div key={c} className="p-1.5 text-[#94A3B8] font-bold truncate text-center" title={c}>
                {c.slice(0, 4)}
              </div>
            ))}

            {metrics.confusion_matrix.map((row, rIdx) => (
              <React.Fragment key={rIdx}>
                <div className="p-1.5 text-slate-300 font-bold truncate self-center" title={classes[rIdx]}>
                  {classes[rIdx].slice(0, 4)}
                </div>
                {row.map((val, cIdx) => {
                  const isDiagonal = rIdx === cIdx;
                  return (
                    <div
                      key={cIdx}
                      className={`p-1.5 rounded-sm text-center font-bold font-mono transition ${
                        isDiagonal
                          ? "bg-indigo-600 text-white"
                          : val > 0
                          ? "bg-[#1E293B] text-slate-300"
                          : "bg-[#0B1120] text-slate-600 border border-[#1E293B]/40"
                      }`}
                      title={`Actual: ${classes[rIdx]}, Predicted: ${classes[cIdx]} (${val} samples)`}
                    >
                      {val}
                    </div>
                  );
                })}
              </React.Fragment>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
