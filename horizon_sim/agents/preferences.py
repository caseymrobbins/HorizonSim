from __future__ import annotations


def normalize_preferences(preferences: dict[str, float]) -> dict[str, float]:
    return {key: max(0.0, min(1.0, float(value))) for key, value in preferences.items()}
