"""
disaster_service.py — Unified CLIP + Qwen2-VL disaster analysis pipeline.

Two-stage workflow:
  Stage 1 — CLIP:   classify the disaster category and confidence score.
  Stage 2 — Qwen:   detailed scene analysis, informed by CLIP's classification.

The CLIP result is injected into the Qwen prompt so Qwen can confirm or
refine the category rather than guessing cold, which improves output quality.

Response schema (returned by run()):
  {
    "category":                 str,   # e.g. "Flood"
    "classification_confidence": float, # from CLIP, 0-100
    "severity":                 str,   # Critical / High / Moderate / Low
    "visible_damage":           str,
    "affected_area":            str,
    "environmental_impact":     str,
    "recommendations":          str,
    "active_models":            list[str],
    "processing_time_ms":       float,
    "clip_raw":                 dict,  # full CLIP output for debugging
    "qwen_raw":                 dict,  # full Qwen output for debugging
  }
"""

from __future__ import annotations

import sys
import time
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Bootstrap sys.path so src/models/ resolves from any working directory.
_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Qwen prompt with CLIP context
# ---------------------------------------------------------------------------

_QWEN_PROMPT = (
    "CLIP classified this image as \"{category}\" ({confidence:.1f}% confidence).\n"
    "Analyze this disaster image.\n"
    "Return ONLY in this exact format:\n"
    "DISASTER TYPE:\n"
    "SEVERITY:\n"
    "ONE-LINE DESCRIPTION:"
)

_FIELD_MAP = {
    "DISASTER TYPE":      "disaster_type",
    "SEVERITY":           "severity",
    "VISIBLE DAMAGE":     "visible_damage",
    "AFFECTED AREA":      "affected_area",
    "ENVIRONMENTAL IMPACT": "environmental_impact",
    "RECOMMENDATIONS":    "recommendations",
}


def _parse_qwen_fields(text: str) -> tuple[dict, float | None]:
    """Parse Qwen's labeled-field output into a structured dict."""
    fields = {v: "" for v in _FIELD_MAP.values()}
    parsed_confidence: float | None = None

    for line in text.splitlines():
        stripped = line.strip()
        for label, key in _FIELD_MAP.items():
            if stripped.startswith(label + ":"):
                fields[key] = stripped[len(label) + 1:].strip()
                break
        if stripped.startswith("CONFIDENCE:"):
            raw_conf = stripped[len("CONFIDENCE:"):].strip()
            try:
                parsed_confidence = float(raw_conf)
            except ValueError:
                pass

    return fields, parsed_confidence


# ---------------------------------------------------------------------------
# CLIP output normaliser
# ---------------------------------------------------------------------------

def _extract_clip(clip_raw: dict) -> tuple[str, float]:
    """
    Extract category and confidence from the CLIP service output.

    Handles both possible schemas:
      - {"metrics": {"disaster_type": ..., "confidence_score": ...}}  (current)
      - {"prediction": ..., "confidence": ...}                         (legacy)
    """
    if "metrics" in clip_raw:
        m = clip_raw["metrics"]
        return (
            m.get("disaster_type", "Unknown"),
            float(m.get("confidence_score", 0)),
        )
    return (
        clip_raw.get("prediction", "Unknown"),
        float(clip_raw.get("confidence", 0)),
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(image_path: str) -> dict:
    """
    Run the two-stage CLIP → Qwen disaster analysis pipeline.

    Args:
        image_path: Absolute path to a temporary image file.

    Returns:
        Unified disaster intelligence report dict.
    """
    t0 = time.perf_counter()

    # ── Stage 1: CLIP ────────────────────────────────────────────────────────
    from models.clip_model import predict_disaster as clip_predict
    clip_raw = clip_predict(image_path)
    category, confidence = _extract_clip(clip_raw)
    log.info(f"CLIP → {category} ({confidence:.1f}%)")

    # ── Stage 2: Qwen with CLIP context ──────────────────────────────────────
    from models.qwen_model import predict_response as qwen_predict
    prompt = _QWEN_PROMPT.format(category=category, confidence=confidence)
    qwen_raw = qwen_predict(image_path, prompt=prompt)

    qwen_text = qwen_raw.get("metrics", {}).get("raw_analysis", "")
    fields, parsed_conf = _parse_qwen_fields(qwen_text)

    # Prefer Qwen's self-reported confidence; fall back to token confidence.
    qwen_confidence = qwen_raw.get("metrics", {}).get("confidence_score", confidence)
    if parsed_conf is not None and 0 <= parsed_conf <= 100:
        qwen_confidence = parsed_conf

    severity    = fields.get("severity") or qwen_raw.get("metrics", {}).get("severity", "Unknown")
    final_type  = fields.get("disaster_type") or category

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    log.info(f"Unified analysis complete in {elapsed_ms} ms — {final_type}, {severity}")

    return {
        "category":                  final_type,
        "classification_confidence": round(confidence, 2),
        "severity":                  severity,
        "visible_damage":            fields.get("visible_damage", ""),
        "affected_area":             fields.get("affected_area", ""),
        "environmental_impact":      fields.get("environmental_impact", ""),
        "recommendations":           fields.get("recommendations", ""),
        "active_models":             ["CLIP", "Qwen2-VL"],
        "processing_time_ms":        elapsed_ms,
        "clip_raw":                  clip_raw,
        "qwen_raw":                  qwen_raw,
    }
