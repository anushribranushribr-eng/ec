export type CryClass =
  | "belly_pain"
  | "burping"
  | "cold_hot"
  | "discomfort"
  | "hungry"
  | "lonely"
  | "scared"
  | "tired"
  | "unknown";

export interface ClassProbability {
  className: CryClass;
  displayName: string;
  probability: number;
  description: string;
  color: string;
}

export interface PredictionResult {
  prediction: CryClass;
  confidence: number;
  rawTopClass: CryClass;
  rawTopConfidence: number;
  isUnknown: boolean;
  confidenceThreshold: number;
  probabilities: Record<CryClass, number>;
  safetyDisclaimer: string;
  filename?: string;
  timestamp?: string;
}

export interface ModelConfig {
  name: string;
  version: string;
  architecture: string;
  backbone: string;
  sample_rate: number;
  channels: number;
  n_fft: number;
  hop_length: number;
  n_mels: number;
  f_min: number;
  f_max: number;
  target_duration_sec: number;
  classes: CryClass[];
  confidence_threshold: number;
  smoothing_alpha: number;
  safety_disclaimer: string;
}

export interface PerClassMetric {
  precision: number;
  recall: number;
  f1_score: number;
  support: number;
}

export interface BenchmarkMetrics {
  accuracy: number;
  macro_f1: number;
  weighted_f1: number;
  total_samples: number;
  unique_subjects: number;
  subject_leakage_percentage: number;
  per_class: Record<CryClass, PerClassMetric>;
  confusion_matrix: number[][];
}

export interface FilenameMetadata {
  filename: string;
  uuid: string;
  timestamp: string;
  version: string;
  gender: string;
  age_months: number;
  raw_reason: string;
  parsed_label: string;
  baby_id: string;
}

export interface SampleCry {
  id: string;
  label: CryClass;
  displayName: string;
  description: string;
  filename: string;
  duration: string;
  gender: "male" | "female";
  ageMonths: number;
}
