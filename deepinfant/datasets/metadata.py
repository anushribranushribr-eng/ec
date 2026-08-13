"""
Dataset Metadata and Filename Parsing Module.
Parses iOS (.caf) and Android (.3gp) infant cry filename patterns and parses CSV metadata.
"""

import os
import re
import pandas as pd
from typing import Dict, Any, Optional

LABEL_MAP = {
    "hu": "hungry",
    "bu": "burping",
    "bp": "belly_pain",
    "dc": "discomfort",
    "ti": "tired",
    "lo": "lonely",
    "sc": "scared",
    "ch": "cold_hot",
    "un": "unknown",
    # Direct class names fallback
    "hungry": "hungry",
    "burping": "burping",
    "belly_pain": "belly_pain",
    "discomfort": "discomfort",
    "tired": "tired",
    "lonely": "lonely",
    "scared": "scared",
    "cold_hot": "cold_hot",
    "unknown": "unknown"
}


def parse_cry_filename(filename: str) -> Dict[str, Any]:
    """
    Parses structured filenames:
    e.g. '0D1AD73E-4C5E-45F3-85C4-9A3CB71E8856-1430742197-1.0-m-04-hu.caf'
    or 'UUID-timestamp-version-gender-age-reason.3gp'
    
    Returns dictionary with parsed fields: uuid, timestamp, version, gender, age_months, reason, baby_id
    """
    basename = os.path.basename(filename)
    name_without_ext, ext = os.path.splitext(basename)
    
    parts = name_without_ext.split("-")
    
    result = {
        "filename": basename,
        "uuid": None,
        "timestamp": None,
        "version": None,
        "gender": "unknown",
        "age_months": None,
        "raw_reason": "unknown",
        "label": "unknown",
        "baby_id": None
    }
    
    # Check if format has enough components
    if len(parts) >= 6:
        # Standard format: UUID (5 segments e.g. 8-4-4-4-12), timestamp, version, gender, age, reason
        # Look at last segments
        reason_code = parts[-1].lower()
        age_str = parts[-2]
        gender_str = parts[-3].lower()
        version_str = parts[-4]
        timestamp_str = parts[-5]
        uuid_str = "-".join(parts[:-5])
        
        result["uuid"] = uuid_str if uuid_str else "UUID_UNKNOWN"
        result["timestamp"] = timestamp_str
        result["version"] = version_str
        result["gender"] = "male" if gender_str in ["m", "male"] else ("female" if gender_str in ["f", "female"] else "unknown")
        
        try:
            result["age_months"] = int(age_str)
        except ValueError:
            result["age_months"] = 0
            
        result["raw_reason"] = reason_code
        result["label"] = LABEL_MAP.get(reason_code, "unknown")
        # Subject ID derived from UUID
        result["baby_id"] = result["uuid"]
    else:
        # Fallback: check if parent folder name or substring matches label
        for code, label in LABEL_MAP.items():
            if f"-{code}." in basename.lower() or f"_{code}." in basename.lower() or code in parts:
                result["label"] = label
                result["raw_reason"] = code
                break
        result["baby_id"] = name_without_ext[:8] if len(name_without_ext) >= 8 else "INFANT_GENERIC"
        
    return result


def load_metadata_csv(csv_path: str) -> pd.DataFrame:
    """Loads CSV metadata containing file paths, labels, and subject IDs."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata CSV file not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = ["filename", "label", "baby_id"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Metadata CSV missing required column: '{col}'")
    return df
