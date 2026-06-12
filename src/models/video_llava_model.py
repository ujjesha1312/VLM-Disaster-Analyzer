"""
video_llava_model.py — Video-LLaVA integration stub.

Video-LLaVA extends the LLaVA architecture to understand video sequences
by encoding multiple uniformly-sampled frames and feeding them into the
language model alongside a text prompt.

Installation (run once in your environment):
    pip install transformers accelerate decord
    # Model will be downloaded from Hugging Face on first load (~14 GB)
    # HF repo: LanguageBind/Video-LLaVA-7B-hf

Paper: "Video-LLaVA: Learning United Visual Representation by Alignment Before Projection"
       https://arxiv.org/abs/2311.10122

Interface is intentionally identical to the existing image models
(clip_model.py, llava_model.py) so it can be slotted into the
existing prediction pipeline with zero changes to the backend router.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class VideoLLaVAModel:
    """
    Wraps the Video-LLaVA-7B model for disaster-scene video analysis.

    Usage:
        model = VideoLLaVAModel()
        model.load()
        result = model.predict("path/to/clip.mp4")
        print(result["response"])          # structured disaster assessment
        print(result["confidence"])        # 0–100 float
    """

    MODEL_ID   = "LanguageBind/Video-LLaVA-7B-hf"
    NUM_FRAMES = 8          # frames uniformly sampled from each clip
    MAX_NEW_TOKENS = 512

    # System prompt tuned for disaster intelligence context
    SYSTEM_PROMPT = (
        "You are a disaster intelligence analyst. "
        "Analyse the video clip and provide a structured assessment including: "
        "(1) Event type and category, "
        "(2) Severity level (Critical / High / Moderate / Low), "
        "(3) Visible damage indicators, "
        "(4) Immediate response priorities, "
        "(5) Any infrastructure or human impact observed. "
        "Be concise, factual, and use operational language."
    )

    def __init__(self, device: str = "auto") -> None:
        self.device    = device
        self.model     = None
        self.processor = None
        self._loaded   = False

    # ── Model loading ─────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load model and processor from Hugging Face Hub (cached after first run)."""
        if self._loaded:
            return
        try:
            from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
            import torch

            log.info(f"Loading Video-LLaVA from {self.MODEL_ID} …")
            self.processor = VideoLlavaProcessor.from_pretrained(self.MODEL_ID)
            self.model     = VideoLlavaForConditionalGeneration.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.float16,
                device_map=self.device,
            )
            self._loaded = True
            log.info("Video-LLaVA loaded successfully.")
        except ImportError as e:
            raise ImportError(
                "Missing dependencies. Install with:\n"
                "  pip install transformers accelerate decord"
            ) from e
        except Exception as e:
            log.error(f"Failed to load Video-LLaVA: {e}")
            raise

    # ── Frame extraction ──────────────────────────────────────────────────────

    def _extract_frames(self, video_path: str) -> Any:
        """
        Extract NUM_FRAMES uniformly-sampled frames from a video file.
        Returns a numpy array of shape (N, H, W, C) in RGB uint8.
        """
        try:
            import decord
            import numpy as np
            decord.bridge.set_bridge("native")
            vr    = decord.VideoReader(video_path, ctx=decord.cpu(0))
            total = len(vr)
            indices = [int(i * (total - 1) / (self.NUM_FRAMES - 1))
                       for i in range(self.NUM_FRAMES)]
            indices = sorted(set(indices))[:self.NUM_FRAMES]  # deduplicate
            frames = vr.get_batch(indices).asnumpy()           # (N, H, W, C)
            return frames
        except ImportError:
            raise ImportError("Install decord: pip install decord")

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, video_path: str, prompt: str | None = None) -> dict:
        """
        Analyse a video clip and return a structured disaster assessment.

        Args:
            video_path : Path to the .mp4 / .avi / .mov clip.
            prompt     : Optional custom question. Defaults to SYSTEM_PROMPT.

        Returns:
            {
                "response"    : str   — free-text assessment,
                "confidence"  : float — estimated confidence 0–100,
                "model"       : str   — model identifier,
                "frames_used" : int   — number of frames sampled,
                "error"       : str | None,
            }
        """
        if not self._loaded:
            self.load()

        if not Path(video_path).exists():
            return self._error_result(f"File not found: {video_path}")

        effective_prompt = prompt or self.SYSTEM_PROMPT

        try:
            import torch
            frames = self._extract_frames(video_path)

            # Build conversation in LLaVA chat template
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": effective_prompt},
                    ],
                }
            ]
            text_prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )

            inputs = self.processor(
                text=text_prompt,
                videos=frames,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.MAX_NEW_TOKENS,
                    do_sample=False,
                )

            # Decode only the newly generated tokens
            generated = output_ids[:, inputs["input_ids"].shape[1]:]
            response  = self.processor.decode(generated[0], skip_special_tokens=True).strip()

            # Heuristic confidence from response length (placeholder)
            # Replace with a dedicated scoring head in production
            confidence = min(95.0, 50.0 + len(response.split()) * 0.4)

            return {
                "response":    response,
                "confidence":  round(confidence, 1),
                "model":       "video-llava-7b",
                "frames_used": len(frames),
                "error":       None,
            }

        except Exception as exc:
            log.error(f"Video-LLaVA inference error: {exc}")
            return self._error_result(str(exc))

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _error_result(message: str) -> dict:
        return {
            "response":    "",
            "confidence":  0.0,
            "model":       "video-llava-7b",
            "frames_used": 0,
            "error":       message,
        }


# ---------------------------------------------------------------------------
# Module-level singleton (mirrors existing model files)
# ---------------------------------------------------------------------------

_model: VideoLLaVAModel | None = None


def get_model() -> VideoLLaVAModel:
    global _model
    if _model is None:
        _model = VideoLLaVAModel()
    return _model


def predict(video_path: str, prompt: str | None = None) -> dict:
    """Convenience function matching the interface of image model modules."""
    return get_model().predict(video_path, prompt=prompt)
