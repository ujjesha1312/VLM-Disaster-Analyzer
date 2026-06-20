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

import asyncio
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
    "DISASTER TYPE: <type>\n"
    "SEVERITY: <critical/high/moderate/low>\n"
    "DESCRIPTION: <one sentence>\n"
    "VISIBLE DAMAGE: <one sentence about physical damage visible in image>\n"
    "AFFECTED AREA: <one sentence about geographic/structural scope>\n"
    "ENVIRONMENTAL IMPACT: <one sentence about environmental consequence>\n"
    "RECOMMENDATIONS: <one sentence on immediate action needed>"
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
# Fix E — Severity normalisation
# ---------------------------------------------------------------------------

# Canonical severity levels expected by the frontend severityChipClass().
_SEVERITY_CANONICAL = {"Critical", "High", "Moderate", "Low"}

# Exact-match lookup (case-insensitive) to a canonical level.
_SEVERITY_MAP: dict[str, str] = {
    # canonical forms
    "critical":  "Critical",
    "high":      "High",
    "moderate":  "Moderate",
    "low":       "Low",
    # synonyms / alternate wordings Qwen may produce
    "severe":    "Critical",
    "very high": "Critical",
    "extreme":   "Critical",
    "serious":   "High",
    "medium":    "Moderate",
    "mild":      "Low",
    "minor":     "Low",
    "minimal":   "Low",
    "unknown":   "Moderate",
}


def _normalize_severity(raw: str) -> str:
    """
    Normalize Qwen's raw severity string to one of:
    Critical / High / Moderate / Low.

    Handles case variations, synonyms, and freeform phrases (e.g. "HIGH risk").
    Defaults to "Moderate" for anything unrecognised.
    """
    if not raw:
        return "Moderate"

    # Direct lookup (case-insensitive)
    lower = raw.strip().lower()
    if lower in _SEVERITY_MAP:
        return _SEVERITY_MAP[lower]

    # Substring scan — handles "HIGH risk" or "CRITICAL — immediate response"
    for token, canonical in _SEVERITY_MAP.items():
        if token in lower:
            return canonical

    log.warning("[Severity] Unrecognised value %r — defaulting to Moderate", raw)
    return "Moderate"


# ---------------------------------------------------------------------------
# Fix F — Field defaults
# ---------------------------------------------------------------------------

_FIELD_DEFAULTS: dict[str, str] = {
    "visible_damage":       "Physical damage assessment not available — review imagery directly.",
    "affected_area":        "Geographic scope not assessed — on-ground survey required.",
    "environmental_impact": "Environmental impact assessment pending field evaluation.",
    "recommendations":      "Deploy assessment teams and establish incident command immediately.",
}


def _apply_field_defaults(fields: dict) -> dict:
    """
    Replace any empty string field with a meaningful placeholder.
    Prevents the frontend from rendering blank ReportCard cells.
    """
    for key, default in _FIELD_DEFAULTS.items():
        if not fields.get(key, "").strip():
            fields[key] = default
    return fields


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
# Disaster relevance gate (runs after CLIP, before Qwen)
# ---------------------------------------------------------------------------

# CLIP labels that indicate a non-disaster scene.
_NON_DISASTER_LABELS: frozenset[str] = frozenset({
    "Forest", "Buildings and Street", "Sea", "Human"
})

# Minimum CLIP confidence for the top disaster label to proceed to Qwen.
# Below this threshold the classification is too uncertain to warrant Qwen.
_MIN_DISASTER_CONFIDENCE: float = 20.0


def _check_disaster_relevance(clip_raw: dict) -> tuple[bool, str, float]:
    """
    Return (is_relevant, top_label, top_confidence).

    is_relevant is False when the image is confidently non-disaster
    (top label is in _NON_DISASTER_LABELS) OR when the model is
    uncertain about everything (confidence < _MIN_DISASTER_CONFIDENCE).
    """
    m = clip_raw.get("metrics", {})
    top_label = m.get("disaster_type", "")
    top_conf  = float(m.get("confidence_score", 0))
    if top_label in _NON_DISASTER_LABELS or top_conf < _MIN_DISASTER_CONFIDENCE:
        return False, top_label, top_conf
    return True, top_label, top_conf


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run(image_path: str) -> dict:
    """
    Run the two-stage CLIP → Qwen disaster analysis pipeline.

    Args:
        image_path: Absolute path to a temporary image file.

    Returns:
        Unified disaster intelligence report dict.
    """
    from backend.services.gpu_queue import run_with_gpu_lock

    t0 = time.perf_counter()

    # ── Stage 1: CLIP ────────────────────────────────────────────────────────
    from models.clip_model import predict_disaster as clip_predict
    clip_raw = await run_with_gpu_lock(
        asyncio.to_thread(clip_predict, image_path),
        "CLIP",
    )
    category, confidence = _extract_clip(clip_raw)
    log.info(f"CLIP → {category} ({confidence:.1f}%)")

    # ── Relevance gate — skip Qwen for non-disaster scenes ───────────────────
    is_relevant, _top_label, _top_conf = _check_disaster_relevance(clip_raw)
    if not is_relevant:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        log.info(
            "[Relevance] Non-disaster — top=%r conf=%.1f%% skipping Qwen (%.0f ms)",
            _top_label, _top_conf, elapsed_ms,
        )
        return {
            "status":             "non_disaster",
            "message":            "The uploaded image does not appear to depict a disaster scene.",
            "category":           _top_label,
            "confidence":         round(_top_conf, 2),
            "processing_time_ms": elapsed_ms,
        }

    # ── Stage 2: Qwen with CLIP context ──────────────────────────────────────
    from models.qwen_model import predict_response as qwen_predict
    prompt = _QWEN_PROMPT.format(category=category, confidence=confidence)
    qwen_raw = await run_with_gpu_lock(
        asyncio.to_thread(qwen_predict, image_path, prompt),
        "Qwen",
    )

    qwen_text = qwen_raw.get("metrics", {}).get("raw_analysis", "")
    fields, parsed_conf = _parse_qwen_fields(qwen_text)

    # Prefer Qwen's self-reported confidence; fall back to token confidence.
    qwen_confidence = qwen_raw.get("metrics", {}).get("confidence_score", confidence)
    if parsed_conf is not None and 0 <= parsed_conf <= 100:
        qwen_confidence = parsed_conf

    # Fix E — normalise severity before it reaches the frontend
    raw_severity = (
        fields.get("severity")
        or qwen_raw.get("metrics", {}).get("severity", "")
    )
    severity   = _normalize_severity(raw_severity)
    final_type = fields.get("disaster_type") or category

    # Fix F — apply default text to any empty fields
    fields = _apply_field_defaults(fields)

    # ── Stage 3: Historical retrieval (best-effort, skipped when disabled) ──────
    similar_events:    list[dict] = []
    retrieval_status:  str = "ok"
    retrieval_message: str = ""

    from backend.config import ENABLE_RETRIEVAL
    if ENABLE_RETRIEVAL:
        try:
            from retrieval.search import find_similar_events

            # The FAISS index contains only these three categories.
            # Any disaster type not mapping here is considered unsupported —
            # we return no events rather than mislead with unrelated results.
            _SUPPORTED_CATEGORIES = {"flood", "cyclone", "earthquake"}

            _CATEGORY_MAP: dict[str, str] = {
                "water disaster":        "flood",
                "flood":                 "flood",
                "flooding":              "flood",
                "cyclone":               "cyclone",
                "hurricane":             "cyclone",
                "typhoon":               "cyclone",
                "tropical storm":        "cyclone",
                "storm":                 "cyclone",
                "earthquake":            "earthquake",
                "infrastructure damage": "earthquake",
                "seismic":               "earthquake",
                "human damage":          "earthquake",
                "building damage":       "earthquake",
                "structural damage":     "earthquake",
            }

            cat_filter   = _CATEGORY_MAP.get(final_type.lower())
            is_supported = cat_filter in _SUPPORTED_CATEGORIES

            log.info(
                "[Retrieval] detected=%r  mapped=%r  supported=%s",
                final_type, cat_filter, is_supported,
            )

            if not is_supported:
                # Drought, Wildfire, Landslide, Tsunami, Volcanic Eruption, etc.
                # are not in the index — skip retrieval entirely.
                log.info(
                    "[Retrieval] Skipping — %r is not a supported retrieval category. "
                    "Returning empty events to avoid misleading results.",
                    final_type,
                )
                retrieval_status  = "unsupported_category"
                retrieval_message = "No historical events available for this disaster category."
            else:
                similar_events = await run_with_gpu_lock(
                    asyncio.to_thread(find_similar_events, image_path, 5, cat_filter),
                    "Retrieval-CLIP",
                )
                log.info(
                    "[Retrieval] Filtered search (%r): %d events returned",
                    cat_filter, len(similar_events),
                )

        except Exception as _retrieval_err:
            log.warning("[Retrieval] Skipped due to error: %s", _retrieval_err)
            retrieval_status  = "error"
            retrieval_message = "Retrieval unavailable."

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
        "similar_events":            similar_events,
        "retrieval_status":          retrieval_status,
        "retrieval_message":         retrieval_message,
        "active_models":             ["Disaster Intelligence Engine"],
        "processing_time_ms":        elapsed_ms,
        "clip_raw":                  clip_raw,
        "qwen_raw":                  qwen_raw,
    }
