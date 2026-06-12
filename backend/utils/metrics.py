def confidence_to_level(score: float) -> str:
    if score > 88:
        return "Critical"
    if score > 75:
        return "High"
    if score > 60:
        return "Moderate"
    return "Low"
