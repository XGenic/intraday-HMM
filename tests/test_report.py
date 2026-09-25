import pandas as pd
import pytest

from regime_retrieval.reporting.report import _conclusion


def evidence(direction=-1):
    return pd.DataFrame(
        [
            {
                "fold": fold,
                "horizon_minutes": horizon,
                "status": "ok",
                "mean_difference": direction * 0.01,
                "ci_lower": direction * 0.01 - 0.002,
                "ci_upper": direction * 0.01 + 0.002,
            }
            for fold in range(4)
            for horizon in (5, 15)
        ]
    )


@pytest.mark.parametrize("direction, outcome", [(-1, "improved"), (1, "not_improved")])
def test_directional_conclusion_requires_consistent_complete_evidence(direction, outcome):
    assert _conclusion(evidence(direction), [5, 15], False, 3)["outcome"] == outcome


@pytest.mark.parametrize("problem", ["missing_fold", "interval_crosses_zero", "fold_reversal"])
def test_incomplete_or_unstable_evidence_cannot_claim_direction(problem):
    frame = evidence()
    if problem == "missing_fold":
        frame = frame[frame.fold != 3]
    elif problem == "interval_crosses_zero":
        frame.loc[(frame.fold == 0) & (frame.horizon_minutes == 15), "ci_upper"] = 0.001
    else:
        frame.loc[(frame.fold == 3) & (frame.horizon_minutes == 15), "mean_difference"] = 0.001
    assert _conclusion(frame, [5, 15], False, 3)["outcome"] == "inconclusive"


def test_synthetic_evidence_never_yields_market_skill_conclusion():
    assert _conclusion(evidence(), [5, 15], True, 3)["outcome"] == "synthetic"
