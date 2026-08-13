import { FilenameMetadata } from "../types";

export function parseCryFilenameClient(filename: string): FilenameMetadata {
  const name = filename.trim();
  const basename = name.split("/").pop() || name;
  const nameWithoutExt = basename.replace(/\.[^/.]+$/, "");
  const parts = nameWithoutExt.split("-");

  const labelMap: Record<string, string> = {
    hu: "hungry",
    bu: "burping",
    bp: "belly_pain",
    dc: "discomfort",
    ti: "tired",
    lo: "lonely",
    sc: "scared",
    ch: "cold_hot",
    un: "unknown"
  };

  const result: FilenameMetadata = {
    filename: basename,
    uuid: "UUID_UNKNOWN",
    timestamp: "N/A",
    version: "1.0",
    gender: "unknown",
    age_months: 0,
    raw_reason: "unknown",
    parsed_label: "unknown",
    baby_id: "BABY_UNKNOWN"
  };

  if (parts.length >= 6) {
    const reasonCode = parts[parts.length - 1].toLowerCase();
    const ageStr = parts[parts.length - 2];
    const genderStr = parts[parts.length - 3].toLowerCase();
    const versionStr = parts[parts.length - 4];
    const timestampStr = parts[parts.length - 5];
    const uuidStr = parts.slice(0, parts.length - 5).join("-");

    result.uuid = uuidStr || "UUID_UNKNOWN";
    result.timestamp = timestampStr;
    result.version = versionStr;
    result.gender = genderStr === "m" ? "male" : genderStr === "f" ? "female" : "unknown";
    result.age_months = parseInt(ageStr, 10) || 0;
    result.raw_reason = reasonCode;
    result.parsed_label = labelMap[reasonCode] || reasonCode;
    result.baby_id = result.uuid;
  } else {
    for (const [code, label] of Object.entries(labelMap)) {
      if (basename.toLowerCase().includes(label) || basename.toLowerCase().includes(`-${code}.`)) {
        result.parsed_label = label;
        result.raw_reason = code;
        break;
      }
    }
    result.baby_id = nameWithoutExt.slice(0, 8) || "BABY_GENERIC";
  }

  return result;
}
