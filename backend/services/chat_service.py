"""
chat_service.py — Context-aware disaster chat response engine.

Uses GPT-4o (text-only) when OPENAI_API_KEY is set, with a structured
template-based fallback when no key is available or the API call fails.

The DisasterContext supplied by the frontend contains everything extracted
during the initial multi-model image analysis. The chatbot uses this
pre-extracted context — no image is re-sent on follow-up questions.

Context keys expected:
    eventType     : str   — e.g. "Wildfire"
    confidence    : float — CLIP confidence %
    caption       : str   — BLIP-2 output
    reasoning     : str   — LLaVA output
    sceneAnalysis : str   — Qwen2-VL output
    severity      : str   — "Critical" | "High" | "Moderate" | "Low"
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Template knowledge base (fallback when no OpenAI key)
# ---------------------------------------------------------------------------

_IMPACTS: dict[str, list[str]] = {
    "Flood":      ["Road and transport infrastructure damage", "Residential and commercial property flooding", "Contamination of water supply systems", "Population displacement", "Agricultural and soil damage"],
    "Fire":       ["Widespread vegetation and forest destruction", "Air quality degradation from smoke", "Wildlife habitat and biodiversity loss", "Structural damage to nearby buildings", "Ash and debris water contamination"],
    "Earthquake": ["Structural collapse of buildings and bridges", "Utility disruption (power, gas, water)", "Secondary landslide and aftershock risk", "Population displacement and casualties", "Long-term geological instability"],
    "Landslide":  ["Transportation corridor blockage", "Burial or structural damage to structures", "Drainage and water system disruption", "Secondary flooding in downstream valleys", "Terrain and slope destabilisation"],
    "Cyclone":    ["Wind and storm surge damage to coastal infrastructure", "Widespread power and communication outages", "Low-lying area flooding", "Agricultural destruction across the region", "Debris hazards for responders"],
}

_ACTIONS: dict[str, list[str]] = {
    "Flood":      ["Deploy water rescue teams and inflatable vessels", "Establish elevated evacuation routes and emergency shelters", "Coordinate drainage authorities to manage water levels", "Pre-position pumping equipment in critical zones", "Issue emergency broadcast communications"],
    "Fire":       ["Mobilise aerial and ground fire suppression units", "Establish firebreaks to contain spread", "Order evacuation within fire perimeter", "Activate public health advisories for air quality", "Deploy forward medical units for respiratory and burn treatment"],
    "Earthquake": ["Activate urban search-and-rescue operations", "Deploy structural engineers for rapid building assessment", "Establish emergency medical triage centres", "Inspect and isolate gas and utility lines", "Issue aftershock warnings to the public"],
    "Landslide":  ["Close and reroute all roads in the slide corridor", "Conduct geotechnical assessment for further slide risk", "Evacuate settlements in the debris flow path", "Deploy heavy machinery for access route clearance", "Install slope monitoring sensors"],
    "Cyclone":    ["Activate coastal evacuation protocols", "Pre-position emergency shelters and backup power", "Suspend maritime and aviation operations", "Issue emergency broadcasts for public safety", "Conduct rapid damage assessment post-storm"],
}

_RESOURCES: dict[str, str] = {
    "Flood":      "Water pumps, inflatable rescue boats, life vests, emergency generators, water purification units, temporary shelter materials.",
    "Fire":       "Aerial tankers, ground fire crews, thermal imaging cameras, protective breathing equipment, medical oxygen and burn units.",
    "Earthquake": "Heavy rescue equipment, K9 search-and-rescue units, structural shoring materials, medical trauma kits, satellite communication relays.",
    "Landslide":  "Excavators, slope stabilisation equipment, geotechnical sensors, drainage clearing machinery, slope monitoring devices.",
    "Cyclone":    "Reinforced storm shelters, backup generators, potable water reserves, medical supplies, satellite communication systems.",
}


def _template_response(question: str, ctx: dict) -> str:
    """Generate a structured response without an LLM."""
    q          = question.lower()
    event      = ctx.get("eventType", "disaster")
    confidence = ctx.get("confidence", 0)
    severity   = ctx.get("severity", "Unknown")
    caption    = ctx.get("caption", "")
    reasoning  = ctx.get("reasoning", "")
    scene      = ctx.get("sceneAnalysis", "")
    impacts    = _IMPACTS.get(event, ["Structural and environmental damage", "Population impact", "Infrastructure disruption"])
    actions    = _ACTIONS.get(event, ["Deploy emergency response teams", "Establish incident command", "Prioritise civilian evacuation"])
    resources  = _RESOURCES.get(event, "Multi-agency emergency equipment, medical supplies, and communication systems.")

    if any(w in q for w in ["sever", "how bad", "intensity", "serious", "extent", "danger"]):
        return (
            f"Based on visual analysis, this **{event}** event is assessed as **{severity}** severity "
            f"(classification confidence: **{confidence}%**).\n\n"
            f"{reasoning.split('.')[0] + '.' if reasoning else ''} "
            f"{'Scene assessment confirms: ' + scene.split('.')[0] + '.' if scene else ''}"
        )

    if any(w in q for w in ["emergency", "response", "action", "protocol", "what should", "what to do", "help"]):
        formatted_actions = "\n".join(f"• {a}" for a in actions)
        return (
            f"**Emergency response protocol for {severity.lower()} {event}:**\n\n"
            f"{formatted_actions}\n\n"
            f"{'Current scene context: ' + scene.split('.')[0] + '.' if scene else ''}"
        )

    if any(w in q for w in ["people", "risk", "casualt", "injur", "life", "human", "death", "survivor"]):
        risk_level = "HIGH" if confidence > 80 else "MODERATE"
        return (
            f"{reasoning.split('.')[0] + '. ' if reasoning else ''}"
            f"Given the **{severity.lower()} {event}** scenario, civilian risk is assessed as **{risk_level}**. "
            f"{'Visual evidence: ' + caption + '. ' if caption else ''}"
            f"Immediate search-and-rescue operations and medical staging should be prioritised."
        )

    if any(w in q for w in ["resource", "deploy", "equipment", "supply", "material", "asset", "send"]):
        return (
            f"**Recommended resources for {severity.lower()} {event}:**\n\n"
            f"{resources}\n\n"
            f"Scale deployment against the **{confidence}%** confidence assessment."
        )

    if any(w in q for w in ["infrastructure", "road", "build", "damage", "structur", "bridge", "power", "utility"]):
        return (
            f"{scene.split(chr(10))[0] if scene else ''} "
            f"{'Visual description: ' + caption + '.' if caption else ''}\n\n"
            f"Priority infrastructure assessment should cover: transportation networks, "
            f"utility systems (power, water, gas), emergency service access routes, "
            f"and critical communication infrastructure."
        )

    if any(w in q for w in ["environment", "ecology", "wildlife", "vegetation", "nature"]):
        env_impacts = [i for i in impacts if any(w in i.lower() for w in ["vegetation", "soil", "water", "habitat", "ecology"])]
        formatted = "\n".join(f"• {i}" for i in (env_impacts or impacts[:3]))
        return (
            f"**Environmental impacts of this {event} event:**\n\n"
            f"{formatted}\n\n"
            f"Long-term ecological monitoring and environmental remediation planning "
            f"should commence alongside immediate emergency response."
        )

    if any(w in q for w in ["model", "confidence", "caption", "predict", "analysis", "classif"]):
        return (
            f"**Disaster intelligence analysis summary:**\n\n"
            f"• **Event**: {event} ({confidence}% confidence)\n"
            f"• **Scene description**: {caption or '—'}\n"
            f"• **Scene reasoning**: {reasoning.split('.')[0] + '.' if reasoning else '—'}\n"
            f"• **Structured assessment**: {scene.split('.')[0] + '.' if scene else '—'}"
        )

    # Default — synthesise available context
    return (
        f"Regarding this **{event}** event ({confidence}% confidence, severity: {severity}):\n\n"
        f"{reasoning.split('.')[0] + '. ' if reasoning else ''}"
        f"{scene.split('.')[0] + '. ' if scene else ''}\n\n"
        f"I can provide specific analysis on severity assessment, emergency protocols, "
        f"infrastructure damage, resource requirements, human risk, or environmental impact. "
        f"What would you like to explore?"
    )


# ---------------------------------------------------------------------------
# GPT-4o powered response (when OPENAI_API_KEY is set)
# ---------------------------------------------------------------------------

def _build_system_prompt(ctx: dict) -> str:
    return (
        "You are a senior disaster intelligence analyst providing real-time support "
        "to an emergency operations team.\n\n"
        "The following disaster context was extracted from a field photograph by the "
        "disaster intelligence system:\n\n"
        f"• **Event type**: {ctx.get('eventType', 'Unknown')} "
        f"({ctx.get('confidence', 0):.1f}% classification confidence)\n"
        f"• **Visual description**: {ctx.get('caption', 'Not available')}\n"
        f"• **Scene reasoning**: {ctx.get('reasoning', 'Not available')}\n"
        f"• **Structured assessment**: {ctx.get('sceneAnalysis', 'Not available')}\n"
        f"• **Assessed severity**: {ctx.get('severity', 'Unknown')}\n\n"
        "Answer user questions using only this pre-extracted context — do NOT request "
        "the image again. Be precise, professional, and actionable. Use domain-specific "
        "emergency management terminology. Keep responses under 200 words."
    )


def generate_response(question: str, ctx: dict, history: list[dict]) -> str:
    """
    Generate a chat response.

    Tries GPT-4o first (text-only, using pre-extracted context as system message).
    Falls back to the structured template engine if the API call fails or the key
    is absent.

    Args:
        question : The user's current question.
        ctx      : DisasterContext object from the client.
        history  : Recent chat messages [{"role": ..., "content": ...}].

    Returns:
        str — the assistant's reply.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            messages = [{"role": "system", "content": _build_system_prompt(ctx)}]

            # Include last 8 exchanges for conversational context
            for msg in history[-8:]:
                role = msg.get("role", "user")
                if role in ("user", "assistant"):
                    messages.append({"role": role, "content": msg.get("content", "")})

            messages.append({"role": "user", "content": question})

            completion = client.chat.completions.create(
                model=os.getenv("GPT4V_MODEL", "gpt-4o"),
                messages=messages,
                max_tokens=400,
                temperature=0.2,
            )
            return completion.choices[0].message.content.strip()

        except Exception as exc:
            logger.warning("GPT-4o chat call failed, falling back to template: %s", exc)

    return _template_response(question, ctx)
