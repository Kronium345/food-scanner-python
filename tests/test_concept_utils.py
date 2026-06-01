"""Tests for concept post-processing."""

from app.services.concept_utils import build_predict_concepts


def test_build_predict_concepts_filters_low_confidence() -> None:
    raw = [
        {"name": "apple pie", "confidence": 0.88},
        {"name": "donuts", "confidence": 0.03},
        {"name": "pizza", "confidence": 0.12},
    ]
    primary, filtered = build_predict_concepts(raw, min_confidence=0.15)

    assert primary is not None
    assert primary.name == "apple pie"
    assert primary.confidence == 0.88
    assert len(filtered) == 1
    assert filtered[0].name == "apple pie"


def test_build_predict_concepts_empty_raw() -> None:
    primary, filtered = build_predict_concepts([], min_confidence=0.15)
    assert primary is None
    assert filtered == []
