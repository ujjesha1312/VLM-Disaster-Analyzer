"""
gpt4v_model.py — Provider-abstracted GPT-4o Vision layer for disaster image analysis.

Architecture:
  This module implements a provider abstraction pattern rather than coupling
  directly to a single vendor API. This enables future extension to additional
  cloud vision providers (Azure OpenAI, Anthropic Claude, Google Gemini) without
  changing the route or service layers.

  ┌─────────────────────────────────────────────────────────┐
  │                   VisionProvider (ABC)                  │
  │  analyze(image_path, prompt) → str                      │
  │  provider_name → str                                    │
  └─────────────────┬───────────────────────────────────────┘
                    │ implements
        ┌───────────┴────────────┐
        ▼                        ▼  (future)
  OpenAIVisionProvider     AzureOpenAIVisionProvider
  gpt-4o / gpt-4-vision    azure-gpt-4-vision
                            ...etc

  get_provider() → VisionProvider   (factory, reads env vars)
  predict_disaster(image_path) → dict  (public API called by service layer)

Configuration:
  OPENAI_API_KEY  — required for OpenAI provider
  GPT4V_MODEL     — optional, defaults to "gpt-4o" (overridable via .env)

Inference flow:
  Image path → base64-encode → OpenAI Chat Completions API (vision)
  → parse labeled fields → return structured metrics dict
"""

import base64
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a senior disaster assessment analyst with expertise in natural hazards, "
    "remote sensing imagery, and emergency response classification. "
    "Your assessments are used by emergency responders and disaster relief coordinators. "
    "When shown a disaster photograph, deliver a precise, structured field report using "
    "domain-specific terminology. Be factual, concise, and avoid speculation beyond "
    "what is visually evident."
)

_USER_PROMPT = (
    "Analyze this disaster scene photograph and provide a structured assessment "
    "using ONLY these labeled sections:\n\n"
    "DISASTER TYPE: [specific type, e.g., riverine flood, flash flood, wildfire, "
    "structural earthquake damage, debris flow landslide, tropical cyclone]\n\n"
    "SEVERITY: [Critical / High / Moderate / Low — one-sentence justification]\n\n"
    "KEY OBSERVATIONS:\n"
    "- [visual observation 1]\n"
    "- [visual observation 2]\n"
    "- [visual observation 3]\n\n"
    "RECOMMENDED ACTIONS:\n"
    "- [action 1]\n"
    "- [action 2]\n"
    "- [action 3]\n\n"
    "Respond with ONLY these four labeled sections. Do not add additional headers "
    "or narrative paragraphs outside the labeled fields."
)

_MIME_MAP: dict[str, str] = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".bmp":  "image/bmp",
    ".webp": "image/webp",
    ".tiff": "image/tiff",
}


# ---------------------------------------------------------------------------
# Response Parser
# ---------------------------------------------------------------------------

def _parse_response(text: str) -> dict:
    """Parse GPT-4V structured response into fields."""
    disaster_type     = ""
    severity          = ""
    confidence_level  = ""
    key_observations  = []
    recommended_actions = []

    mode = None

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("DISASTER TYPE:"):
            disaster_type = stripped[len("DISASTER TYPE:"):].strip()
            mode = None
        elif stripped.startswith("SEVERITY:"):
            severity_full = stripped[len("SEVERITY:"):].strip()
            for level in ("Critical", "High", "Moderate", "Low"):
                if severity_full.lower().startswith(level.lower()):
                    severity         = level
                    confidence_level = level
                    break
            mode = None
        elif stripped.startswith("KEY OBSERVATIONS:"):
            mode = "observations"
        elif stripped.startswith("RECOMMENDED ACTIONS:"):
            mode = "actions"
        elif stripped.startswith("-"):
            item = stripped[1:].strip()
            if item and mode == "observations":
                key_observations.append(item)
            elif item and mode == "actions":
                recommended_actions.append(item)
        elif stripped and not stripped.startswith("-"):
            # Non-bullet content outside a list section resets the list mode
            if mode not in ("observations", "actions"):
                mode = None

    return {
        "disaster_type":      disaster_type,
        "severity":           severity,
        "confidence_level":   confidence_level,
        "key_observations":   key_observations,
        "recommended_actions": recommended_actions,
    }


# ---------------------------------------------------------------------------
# Abstract Provider Base
# ---------------------------------------------------------------------------

class VisionProvider(ABC):
    """Abstract base class for cloud vision API providers.

    All concrete providers must implement analyze() and provider_name.
    This decouples the inference pipeline from any specific vendor.
    """

    @abstractmethod
    def analyze(self, image_path: str, system_prompt: str, user_prompt: str) -> str:
        """Send an image and prompts to the provider; return the text response.

        Args:
            image_path:    Absolute path to the image file.
            system_prompt: System-level instruction for the model.
            user_prompt:   User-turn question or instruction.

        Returns:
            Raw text response from the vision model.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider + model identifier (used in response metadata)."""
        ...


# ---------------------------------------------------------------------------
# OpenAI Vision Provider
# ---------------------------------------------------------------------------

class OpenAIVisionProvider(VisionProvider):
    """GPT-4o Vision via the OpenAI Chat Completions API.

    Encodes the image as base64 and sends it in the 'image_url' content block
    using the data URI scheme — no file hosting required.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self._api_key = api_key
        self._model   = model

    @property
    def provider_name(self) -> str:
        return f"OpenAI / {self._model}"

    # -- Image encoding -------------------------------------------------------

    @staticmethod
    def _encode_image(path: Path) -> tuple[str, str]:
        """Return (base64_data_string, mime_type) for a given image path."""
        mime = _MIME_MAP.get(path.suffix.lower(), "image/jpeg")
        with path.open("rb") as fh:
            return base64.b64encode(fh.read()).decode("utf-8"), mime

    # -- Core inference -------------------------------------------------------

    def analyze(self, image_path: str, system_prompt: str, user_prompt: str) -> str:
        """Send image + prompts to GPT-4o Vision and return the text response."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The openai package is required for GPT-4V inference. "
                "Install it with: pip install openai"
            ) from exc

        path = Path(image_path)
        b64_data, mime = self._encode_image(path)
        data_uri = f"data:{mime};base64,{b64_data}"

        client = OpenAI(api_key=self._api_key)
        logger.info(
            "Calling %s with image '%s' (%.1f KB) ...",
            self._model, path.name, path.stat().st_size / 1024,
        )

        completion = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                },
            ],
            max_tokens=400,
            temperature=0,   # deterministic for reproducible analysis
        )

        response_text = completion.choices[0].message.content.strip()
        logger.info("GPT-4o response received (%d chars).", len(response_text))
        return response_text


# ---------------------------------------------------------------------------
# Provider Factory
# ---------------------------------------------------------------------------

def get_provider() -> VisionProvider:
    """Instantiate and return the configured vision provider.

    Currently supports OpenAI. Extend this function to support additional
    providers by checking additional environment variables.

    Environment variables:
        OPENAI_API_KEY  — activates the OpenAI provider (required)
        GPT4V_MODEL     — override the model ID (default: gpt-4o)

    Returns:
        A configured VisionProvider instance ready for inference.

    Raises:
        EnvironmentError: If no provider can be configured (missing API keys).
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        model_id = os.getenv("GPT4V_MODEL", "gpt-4o")
        logger.info("Using OpenAI provider with model '%s'.", model_id)
        return OpenAIVisionProvider(api_key=openai_key, model=model_id)

    # ── Future providers can be added here ────────────────────────────────────
    # azure_key = os.getenv("AZURE_OPENAI_KEY")
    # if azure_key:
    #     return AzureOpenAIVisionProvider(api_key=azure_key, ...)
    # ──────────────────────────────────────────────────────────────────────────

    raise EnvironmentError(
        "No vision provider is configured. "
        "Set OPENAI_API_KEY in your .env file to enable GPT-4V inference. "
        "See .env.example for the required format."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_disaster(image_path: str) -> dict:
    """Run GPT-4o Vision analysis on a disaster image.

    Validates the image, selects the configured provider via the factory,
    and returns structured metrics parsed from the model's labeled response.

    Args:
        image_path: Absolute path to the image file.

    Returns:
        {
            "model":    "GPT-4V",
            "provider": str,   # e.g. "OpenAI / gpt-4o"
            "metrics":  {
                "disaster_type":       str,
                "severity":            str,
                "confidence_score":    None,  # GPT-4V has no token scores
                "confidence_level":    str,
                "key_observations":    list[str],
                "recommended_actions": list[str],
                "raw_response":        str,
            }
        }

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError:        If the file is not a valid image.
        EnvironmentError:  If no vision provider is configured (missing API key).
        ImportError:       If the openai package is not installed.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        # Verify the file is a readable image before making an API call.
        Image.open(path).verify()
    except UnidentifiedImageError as exc:
        raise ValueError(f"Not a valid image file: {image_path}") from exc
    except Exception as exc:
        raise ValueError(f"Could not verify image: {image_path} — {exc}") from exc

    # Factory resolves the provider based on available environment variables.
    provider = get_provider()

    response_text = provider.analyze(
        image_path=image_path,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=_USER_PROMPT,
    )

    parsed = _parse_response(response_text)

    return {
        "model":    "GPT-4V",
        "provider": provider.provider_name,
        "metrics":  {
            "disaster_type":       parsed["disaster_type"],
            "severity":            parsed["severity"],
            "confidence_score":    None,
            "confidence_level":    parsed["confidence_level"],
            "key_observations":    parsed["key_observations"],
            "recommended_actions": parsed["recommended_actions"],
            "raw_response":        response_text,
        },
    }
