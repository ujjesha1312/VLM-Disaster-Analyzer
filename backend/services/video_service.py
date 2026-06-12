"""
video_service.py — Video file processing service.

Handles everything between the HTTP layer and the (future) model layer:
  extract_metadata()   — ffprobe stream info; falls back to cv2 or basic stats
  extract_thumbnail()  — ffmpeg middle-frame JPEG; falls back to cv2
  build_mock_analysis()— structured placeholder assessment (no model yet)

Design contract:
  All public functions accept an absolute path string.
  All public functions return plain dicts or None — never raise on fallback.
  Routes are responsible for HTTP errors; this layer is responsible for data.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Thumbnail target dimensions (keeps aspect ratio)
THUMB_WIDTH  = 640
THUMB_HEIGHT = 360


# ---------------------------------------------------------------------------
# ffprobe helpers
# ---------------------------------------------------------------------------

def _ffprobe(path: str) -> dict | None:
    """
    Run ffprobe and return the parsed JSON or None if ffprobe is unavailable.
    """
    cmd = [
        "ffprobe",
        "-v",            "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    return None


def _parse_fps(rate_str: str) -> float:
    """Convert '30000/1001' or '30' to a float fps."""
    try:
        if "/" in rate_str:
            num, den = rate_str.split("/")
            return round(float(num) / float(den), 3)
        return float(rate_str)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def extract_metadata(path: str) -> dict:
    """
    Return a normalised metadata dict for the video at *path*.

    Primary: ffprobe
    Fallback: opencv VideoCapture
    Last resort: file-stat only (always succeeds)
    """
    p      = Path(path)
    size_b = p.stat().st_size if p.exists() else 0

    base = {
        "filename":     p.name,
        "format":       p.suffix.lstrip(".").upper() or "UNKNOWN",
        "size_mb":      round(size_b / (1024 ** 2), 2),
        "duration_s":   0.0,
        "fps":          0.0,
        "width":        0,
        "height":       0,
        "resolution":   "Unknown",
        "total_frames": 0,
        "codec":        "unknown",
        "source":       "file-stat",
    }

    # ── Try ffprobe ────────────────────────────────────────────────────────────
    info = _ffprobe(path)
    if info:
        fmt     = info.get("format", {})
        streams = info.get("streams", [])
        video_s = next((s for s in streams if s.get("codec_type") == "video"), {})

        duration = float(fmt.get("duration") or video_s.get("duration") or 0)
        fps_raw  = video_s.get("r_frame_rate") or video_s.get("avg_frame_rate") or "0"
        fps      = _parse_fps(fps_raw)
        w        = int(video_s.get("width", 0))
        h        = int(video_s.get("height", 0))
        codec    = video_s.get("codec_name", "unknown")

        # Derive total frames
        frames = int(video_s.get("nb_frames") or 0)
        if frames == 0 and fps > 0 and duration > 0:
            frames = int(duration * fps)

        base.update({
            "duration_s":   round(duration, 3),
            "fps":          fps,
            "width":        w,
            "height":       h,
            "resolution":   f"{w}x{h}" if w and h else "Unknown",
            "total_frames": frames,
            "codec":        codec,
            "source":       "ffprobe",
        })
        return base

    # ── Try opencv fallback ────────────────────────────────────────────────────
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            fps      = cap.get(cv2.CAP_PROP_FPS) or 0.0
            frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration = frames / fps if fps > 0 else 0.0
            cap.release()
            base.update({
                "duration_s":   round(duration, 3),
                "fps":          round(fps, 3),
                "width":        w,
                "height":       h,
                "resolution":   f"{w}x{h}" if w and h else "Unknown",
                "total_frames": frames,
                "codec":        "unknown (cv2)",
                "source":       "opencv",
            })
    except ImportError:
        log.debug("opencv not available — using file-stat metadata")
    except Exception as e:
        log.debug(f"cv2 fallback failed: {e}")

    return base


# ---------------------------------------------------------------------------
# Thumbnail extraction
# ---------------------------------------------------------------------------

def extract_thumbnail(path: str, seek_s: float = 1.0) -> str | None:
    """
    Extract a single JPEG frame and return it as a base64 data-URI string.

    Primary: ffmpeg — seek to *seek_s* seconds into the video
    Fallback: cv2 VideoCapture
    Returns None if both methods fail.
    """
    # ── Try ffmpeg ─────────────────────────────────────────────────────────────
    try:
        fd, tmp = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)

        cmd = [
            "ffmpeg",
            "-ss",      str(seek_s),
            "-i",       path,
            "-vframes", "1",
            "-vf",      f"scale={THUMB_WIDTH}:{THUMB_HEIGHT}:force_original_aspect_ratio=decrease"
                        f",pad={THUMB_WIDTH}:{THUMB_HEIGHT}:(ow-iw)/2:(oh-ih)/2,format=yuvj420p",
            "-q:v",     "3",
            "-y",       tmp,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode == 0 and Path(tmp).stat().st_size > 100:
            data = Path(tmp).read_bytes()
            return "data:image/jpeg;base64," + base64.b64encode(data).decode()
        Path(tmp).unlink(missing_ok=True)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    except Exception as e:
        log.debug(f"ffmpeg thumbnail failed: {e}")

    # ── Try cv2 fallback ───────────────────────────────────────────────────────
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
            target_f = int(fps * seek_s)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                frame = cv2.resize(frame, (THUMB_WIDTH, THUMB_HEIGHT),
                                   interpolation=cv2.INTER_AREA)
                ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok2:
                    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    except ImportError:
        log.debug("opencv not available for thumbnail fallback")
    except Exception as e:
        log.debug(f"cv2 thumbnail fallback failed: {e}")

    return None


# ---------------------------------------------------------------------------
# Mock analysis
# ---------------------------------------------------------------------------

def build_mock_analysis(meta: dict, frames_analyzed: int = 8) -> dict:
    """
    Build a placeholder disaster assessment response.

    This is intentionally honest — it clearly communicates that no model has
    run yet while still providing useful metadata-derived information.
    """
    dur  = meta["duration_s"]
    res  = meta["resolution"]
    fps  = meta["fps"]
    fn   = meta["filename"]
    size = meta["size_mb"]
    fmt  = meta["format"]
    tot  = meta["total_frames"]

    # Human-readable duration
    m, s = divmod(int(dur), 60)
    dur_str = f"{m}m {s:02d}s" if m else f"{s}s"

    summary = (
        f"Video file received and processed. Stream analysis complete.\n\n"
        f"• File: {fn} ({size:.1f} MB, {fmt} container)\n"
        f"• Stream: {res} · {fps:.1f} fps · {dur_str} duration · {tot:,} frames\n"
        f"• Codec: {meta['codec']}\n\n"
        f"Visual content classification requires video intelligence model inference. "
        f"Integrate Video-LLaVA for scene description, InternVideo2 for incident "
        f"classification, or Qwen2-VL for detailed multimodal reasoning to generate "
        f"a full disaster assessment from this footage."
    )

    return {
        "event_type":      "Video Assessment",
        "severity":        "Pending",
        "confidence":      0.0,
        "summary":         summary,
        "frames_analyzed": frames_analyzed,
        "assessment_note": "Metadata-only response — video model inference not yet active.",
        "pending_models": [
            {
                "model":       "Video-LLaVA 7B",
                "status":      "pending",
                "endpoint":    "/predict/video/llava",
                "description": "Zero-shot disaster scene description",
            },
            {
                "model":       "InternVideo2 1B",
                "status":      "pending",
                "endpoint":    "/predict/video/internvideo",
                "description": "Temporal incident type classification",
            },
            {
                "model":       "Qwen2-VL 7B",
                "status":      "pending",
                "endpoint":    "/predict/video/qwen",
                "description": "Multimodal video reasoning and Q&A",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Unified video processing pipeline
# ---------------------------------------------------------------------------

def process_video(path: str) -> dict:
    """
    Run the full processing pipeline on a video file.

    Returns the combined response dict ready to send to the client.
    """
    t0 = time.perf_counter()

    meta      = extract_metadata(path)
    thumb_b64 = extract_thumbnail(path)
    analysis  = build_mock_analysis(meta)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "file_info":         meta,
        "thumbnail_b64":     thumb_b64,
        "analysis":          analysis,
        "processing_time_ms": elapsed_ms,
    }
