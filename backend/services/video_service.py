"""
video_service.py — Video file processing service.

Handles everything between the HTTP layer and the model layer:
  extract_metadata()   — ffprobe stream info; falls back to cv2 or basic stats
  extract_thumbnail()  — ffmpeg middle-frame JPEG; falls back to cv2
  extract_frames()     — 4 JPEG frames at 25/50/75/90% of duration for model inference
  _vote_best_frame()   — CLIP on each frame; majority vote + confidence-weighted tiebreak
  process_video()      — full async pipeline (metadata + thumbnail + CLIP + Qwen + FAISS)
                         falls back to metadata-only if models are unavailable

Design contract:
  Sync helpers (extract_metadata, extract_thumbnail, extract_frames) never raise on
  fallback — they return empty/partial values instead.
  process_video is async and always returns a valid response dict.
  Routes are responsible for HTTP errors; this layer is responsible for data.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Bootstrap src/ onto sys.path so model imports resolve from any launch directory.
_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Thumbnail target dimensions (keeps aspect ratio)
THUMB_WIDTH  = 640
THUMB_HEIGHT = 360

# Temporal positions for inference-frame extraction (fraction of total duration)
_FRAME_POSITIONS = (0.25, 0.50, 0.75, 0.90)
# Fixed second offsets used as fallback when video duration is unknown
_FALLBACK_SEEK_S = (1.0,   3.0,  6.0,  10.0)


# ---------------------------------------------------------------------------
# ffprobe helpers
# ---------------------------------------------------------------------------

def _ffprobe(path: str) -> dict | None:
    """Run ffprobe and return the parsed JSON or None if ffprobe is unavailable."""
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
# Thumbnail extraction (display frame only, not used for inference)
# ---------------------------------------------------------------------------

def extract_thumbnail(path: str, seek_s: float = 1.0) -> str | None:
    """
    Extract a single JPEG frame and return it as a base64 data-URI string.

    Primary: ffmpeg — seek to *seek_s* seconds into the video.
    Fallback: cv2 VideoCapture.
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
# Inference frame extraction (4 frames for CLIP + Qwen analysis)
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, duration_s: float) -> list[str]:
    """
    Extract 4 JPEG frames at 25/50/75/90% of video duration for model inference.

    Falls back to fixed second offsets (1, 3, 6, 10 s) when duration is unknown.
    Returns a list of temp JPEG paths. Caller must unlink them after use.
    """
    if duration_s > 0:
        seek_points = [max(0.5, duration_s * f) for f in _FRAME_POSITIONS]
    else:
        seek_points = list(_FALLBACK_SEEK_S)

    paths: list[str] = []

    for seek_s in seek_points:
        fd, tmp = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        extracted = False

        # ── ffmpeg ────────────────────────────────────────────────────────────
        cmd = [
            "ffmpeg",
            "-ss",      f"{seek_s:.3f}",
            "-i",       video_path,
            "-vframes", "1",
            "-vf",      "scale=640:480:force_original_aspect_ratio=decrease",
            "-q:v",     "2",
            "-y",       tmp,
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=30)
            if r.returncode == 0 and Path(tmp).stat().st_size > 100:
                paths.append(tmp)
                extracted = True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        except Exception as e:
            log.debug("ffmpeg frame extract failed at %.1fs: %s", seek_s, e)

        if not extracted:
            # ── cv2 fallback ─────────────────────────────────────────────────
            try:
                import cv2  # type: ignore
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    fps_cv = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(seek_s * fps_cv))
                    ok, frame = cap.read()
                    cap.release()
                    if ok and frame is not None:
                        cv2.imwrite(tmp, frame)
                        paths.append(tmp)
                        extracted = True
            except ImportError:
                log.debug("cv2 not available for frame extraction")
            except Exception as e:
                log.debug("cv2 frame extract failed at %.1fs: %s", seek_s, e)

        if not extracted:
            Path(tmp).unlink(missing_ok=True)

    log.info("[Video] Extracted %d/%d inference frames", len(paths), len(seek_points))
    return paths


# ---------------------------------------------------------------------------
# CLIP majority vote + confidence-weighted tiebreak
# ---------------------------------------------------------------------------

async def _vote_best_frame(
    frame_paths: list[str],
) -> tuple[str, float, str, dict[str, int]]:
    """
    Run CLIP on each frame; select the best frame by majority vote.

    Returns:
        (best_frame_path, best_confidence, winner_category, vote_tally)

    Selection logic:
      1. Count votes per category across all frames.
      2. Tiebreak by summed confidence (higher total wins).
      3. Best frame = highest-confidence frame in the winning category.
    """
    from models.clip_model import predict_disaster as clip_predict
    from backend.services.gpu_queue import run_with_gpu_lock

    frame_results: list[tuple[str, float, str]] = []   # (path, confidence, category)

    for fp in frame_paths:
        try:
            clip_raw = await run_with_gpu_lock(
                asyncio.to_thread(clip_predict, fp),
                "CLIP",
            )
            m    = clip_raw.get("metrics", {})
            cat  = m.get("disaster_type", "Unknown")
            conf = float(m.get("confidence_score", 0.0))
            frame_results.append((fp, conf, cat))
            log.info("[Video] Frame %d/%d → %s (%.1f%%)",
                     len(frame_results), len(frame_paths), cat, conf)
        except Exception as exc:
            log.warning("[Video] CLIP skipped frame %s: %s", fp, exc)

    if not frame_results:
        return frame_paths[0], 0.0, "Unknown", {}

    # Tally votes and confidence sums per category
    vote_count: dict[str, int]   = {}
    conf_sum:   dict[str, float] = {}
    for _, conf, cat in frame_results:
        vote_count[cat] = vote_count.get(cat, 0) + 1
        conf_sum[cat]   = conf_sum.get(cat, 0.0) + conf

    max_votes = max(vote_count.values())
    tied      = [c for c, v in vote_count.items() if v == max_votes]
    winner    = max(tied, key=lambda c: conf_sum[c])

    # Best frame within the winner category = highest individual CLIP confidence
    winner_frames = [(conf, fp) for fp, conf, cat in frame_results if cat == winner]
    best_conf, best_path = max(winner_frames, key=lambda x: x[0])

    log.info("[Video] Winner: %s | best conf: %.1f%% | tally: %s",
             winner, best_conf, vote_count)
    return best_path, best_conf, winner, vote_count


# ---------------------------------------------------------------------------
# Metadata-only fallback assessment
# ---------------------------------------------------------------------------

def build_mock_analysis(meta: dict, frames_analyzed: int = 0) -> dict:
    """
    Placeholder assessment returned when models are unavailable or frame extraction fails.
    Self-describing: clearly communicates that no model ran.
    """
    dur  = meta["duration_s"]
    res  = meta["resolution"]
    fps  = meta["fps"]
    fn   = meta["filename"]
    size = meta["size_mb"]
    fmt  = meta["format"]
    tot  = meta["total_frames"]

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

async def process_video(path: str) -> dict:
    """
    Full video analysis pipeline:
      1. Extract stream metadata + thumbnail (always).
      2. Extract 4 inference frames at 25/50/75/90% of duration.
      3. Run CLIP on each frame; select best by majority vote.
      4. Run disaster_service.run() (CLIP → Qwen2-VL → FAISS) on the best frame.
      5. Return merged response: video_metadata + full disaster intelligence report.

    Falls back to metadata-only response when:
      - CLIP or Qwen are disabled in the deployment profile
      - Frame extraction fails (ffmpeg and cv2 both unavailable)
      - disaster_service raises an unexpected exception
    """
    import inspect
    t0 = time.perf_counter()

    # TRACE: confirm which file this function body was loaded from
    log.warning("[Video:TRACE] process_video() called — loaded from: %s", inspect.getfile(process_video))

    meta      = extract_metadata(path)
    thumb_b64 = extract_thumbnail(path)

    log.warning("[Video:TRACE] metadata extracted — duration_s=%.2f, source=%s",
                meta["duration_s"], meta["source"])

    # Check deployment profile — full video analysis requires CLIP + Qwen
    try:
        from backend.config import ENABLE_CLIP, ENABLE_QWEN
        models_ready = ENABLE_CLIP and ENABLE_QWEN
        log.warning("[Video:TRACE] config — ENABLE_CLIP=%s, ENABLE_QWEN=%s, models_ready=%s",
                    ENABLE_CLIP, ENABLE_QWEN, models_ready)
    except Exception as cfg_exc:
        models_ready = False
        log.warning("[Video:TRACE] config import FAILED: %s — models_ready=False", cfg_exc)

    frame_paths: list[str] = []

    if models_ready:
        try:
            frame_paths = extract_frames(path, meta["duration_s"])
            log.warning("[Video:TRACE] extract_frames returned %d paths: %s",
                        len(frame_paths), frame_paths)
        except Exception as exc:
            log.warning("[Video:TRACE] extract_frames EXCEPTION: %s", exc)
    else:
        log.warning("[Video:TRACE] SKIP extract_frames — models_ready=False")

    if not models_ready:
        log.warning("[Video:TRACE] FALLBACK REASON: models_ready=False")
    elif not frame_paths:
        log.warning("[Video:TRACE] FALLBACK REASON: extract_frames returned 0 frames")

    if models_ready and frame_paths:
        try:
            log.warning("[Video:TRACE] calling _vote_best_frame with %d frames", len(frame_paths))
            best_frame, best_conf, winner_cat, vote_tally = await _vote_best_frame(frame_paths)
            log.warning("[Video:TRACE] _vote_best_frame done — winner=%s conf=%.1f best_frame=%s",
                        winner_cat, best_conf, best_frame)

            log.warning("[Video:TRACE] calling disaster_service.run on best frame")
            from backend.services.disaster_service import run as disaster_run
            disaster = await disaster_run(best_frame)

            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            log.warning("[Video:TRACE] FULL PIPELINE SUCCEEDED — %s %s in %.0f ms",
                        disaster["category"], disaster["severity"], elapsed_ms)

            return {
                # Video stream info
                "video_metadata":   meta,
                "thumbnail_b64":    thumb_b64,
                "frames_analyzed":  len(frame_paths),
                "best_frame_index": frame_paths.index(best_frame),
                "frame_votes":      vote_tally,
                # Disaster intelligence — same field names as /predict/disaster
                "category":                  disaster["category"],
                "disaster_type":             disaster["category"],
                "classification_confidence": disaster["classification_confidence"],
                "severity":                  disaster["severity"],
                "visible_damage":            disaster["visible_damage"],
                "affected_area":             disaster["affected_area"],
                "environmental_impact":      disaster["environmental_impact"],
                "recommendations":           disaster["recommendations"],
                "similar_events":            disaster["similar_events"],
                "active_models":             disaster["active_models"],
                "processing_time_ms":        elapsed_ms,
            }

        except Exception as exc:
            log.exception("[Video:TRACE] PIPELINE EXCEPTION (falling back to metadata): %s", exc)
        finally:
            for fp in frame_paths:
                Path(fp).unlink(missing_ok=True)

    # ── Metadata-only fallback ────────────────────────────────────────────────
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    log.warning("[Video:TRACE] RETURNING METADATA-ONLY RESPONSE (fallback)")
    analysis   = build_mock_analysis(meta, frames_analyzed=len(frame_paths))

    for fp in frame_paths:
        Path(fp).unlink(missing_ok=True)

    return {
        "file_info":          meta,
        "thumbnail_b64":      thumb_b64,
        "analysis":           analysis,
        "processing_time_ms": elapsed_ms,
    }
