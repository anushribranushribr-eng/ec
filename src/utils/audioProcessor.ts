import { CryClass } from "../types";

export const CRY_CLASS_METADATA: Record<
  CryClass,
  { displayName: string; color: string; description: string }
> = {
  hungry: {
    displayName: "Hungry",
    color: "#3B82F6", // Blue
    description: "May correspond to a rhythmic acoustic pattern associated with hunger."
  },
  belly_pain: {
    displayName: "Belly Pain / Gas",
    color: "#EF4444", // Red
    description: "May correspond to an acoustic pattern associated with abdominal discomfort or gas."
  },
  burping: {
    displayName: "Burping Needed",
    color: "#10B981", // Emerald
    description: "May correspond to a pattern associated with needing to burp after feeding."
  },
  discomfort: {
    displayName: "General Discomfort",
    color: "#F59E0B", // Amber
    description: "May correspond to a wet diaper, posture preference, or general discomfort."
  },
  tired: {
    displayName: "Sleepy / Tired",
    color: "#8B5CF6", // Purple
    description: "May correspond to a lower frequency rhythmic vocalization associated with fatigue."
  },
  lonely: {
    displayName: "Needs Affection / Lonely",
    color: "#EC4899", // Pink
    description: "May correspond to a desire for soothing physical closeness."
  },
  scared: {
    displayName: "Startled / Scared",
    color: "#F97316", // Orange
    description: "May correspond to a sudden distress vocalization."
  },
  cold_hot: {
    displayName: "Temperature Discomfort",
    color: "#06B6D4", // Cyan
    description: "May correspond to temperature variation or clothing discomfort."
  },
  unknown: {
    displayName: "Uncertain / Noise",
    color: "#6B7280", // Gray
    description: "The acoustic pattern does not match supported categories with high confidence."
  }
};

/**
 * Draws real-time audio waveform on canvas context
 */
export function drawWaveform(
  ctx: CanvasRenderingContext2D,
  data: Float32Array,
  width: number,
  height: number,
  color: string = "#3B82F6"
) {
  ctx.clearRect(0, 0, width, height);
  
  // Draw subtle grid lines
  ctx.strokeStyle = "rgba(229, 231, 235, 0.4)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();

  const sliceWidth = width / data.length;
  let x = 0;

  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    const y = ((v + 1) / 2) * height;

    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
    x += sliceWidth;
  }

  ctx.stroke();
}

/**
 * Draws 128-bin Mel-Spectrogram Heatmap on canvas
 */
export function drawMelSpectrogramHeatmap(
  ctx: CanvasRenderingContext2D,
  spectrogram: number[][], // [128, timeSteps]
  width: number,
  height: number
) {
  ctx.clearRect(0, 0, width, height);

  const numMels = spectrogram.length; // 128
  if (numMels === 0) return;
  const timeSteps = spectrogram[0].length;
  if (timeSteps === 0) return;

  const cellWidth = width / timeSteps;
  const cellHeight = height / numMels;

  for (let m = 0; m < numMels; m++) {
    // Flip vertical axis so low frequencies are at bottom
    const y = height - (m + 1) * cellHeight;
    for (let t = 0; t < timeSteps; t++) {
      const val = spectrogram[m][t]; // Normalized value between 0.0 and 1.0
      const x = t * cellWidth;

      // Color map: Viridis / Magma style (Dark blue -> Purple -> Pink -> Yellow)
      const hue = (1.0 - val) * 240; // 240 (blue) down to 0 (red)
      const lightness = Math.min(85, Math.max(10, val * 90));

      ctx.fillStyle = `hsl(${hue}, 85%, ${lightness}%)`;
      ctx.fillRect(x, y, cellWidth + 0.5, cellHeight + 0.5);
    }
  }
}

/**
 * Generates synthetic 128-bin Mel-Spectrogram data for canvas rendering
 */
export function generateSyntheticMelSpectrogram(numTimeSteps: number = 80, cryType: CryClass = "hungry"): number[][] {
  const numMels = 128;
  const spec: number[][] = [];

  for (let m = 0; m < numMels; m++) {
    const row: number[] = [];
    // Formant frequency region depending on cry type
    const formantCenter =
      cryType === "hungry" ? 45 :
      cryType === "belly_pain" ? 75 :
      cryType === "burping" ? 30 :
      cryType === "scared" ? 90 : 55;

    for (let t = 0; t < numTimeSteps; t++) {
      const distFromFormant = Math.abs(m - formantCenter);
      const intensity = Math.exp(-distFromFormant / 18.0) * (0.6 + 0.4 * Math.sin(t / 4.0));
      const noise = Math.random() * 0.15;
      const normVal = Math.min(1.0, Math.max(0.0, intensity + noise));
      row.push(normVal);
    }
    spec.push(row);
  }

  return spec;
}
