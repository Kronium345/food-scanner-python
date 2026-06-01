"""Post-process model outputs for Node-friendly consumption."""

from __future__ import annotations

from app.models.schemas import Concept


def build_predict_concepts(
    raw: list[dict[str, float | str]],
    *,
    min_confidence: float,
) -> tuple[Concept | None, list[Concept]]:
    """
    Split raw classifier output into a primary label and a filtered concept list.

    - primaryConcept: always the top model prediction (highest confidence).
    - concepts: predictions with confidence >= min_confidence (drops noise like 3% donuts).
    """
    if not raw:
        return None, []

    ordered = sorted(raw, key=lambda item: float(item["confidence"]), reverse=True)
    primary = Concept(
        name=str(ordered[0]["name"]),
        confidence=float(ordered[0]["confidence"]),
    )

    filtered = [
        Concept(name=str(item["name"]), confidence=float(item["confidence"]))
        for item in ordered
        if float(item["confidence"]) >= min_confidence
    ]

    return primary, filtered
