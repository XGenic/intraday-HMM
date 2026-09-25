"""Training-fitted robust volume normalization by exchange-session minute."""

import numpy as np
import pandas as pd


class VolumeSeasonality:
    """Normalize each slot by its training median and 1.4826 times its MAD.

    A zero MAD uses ``max(abs(training_slot_median), 1)`` as its nonzero
    scale. This deliberately conservative fallback handles constant or singleton
    slots without dividing by epsilon or borrowing inference observations. Slots
    with no finite training volume, including unseen slots, transform to NaN.
    Fitting ignores nonfinite observations; transforming never updates statistics.
    """

    def __init__(self) -> None:
        self.statistics_: pd.DataFrame | None = None

    def fit(self, training: pd.DataFrame) -> "VolumeSeasonality":
        finite = np.isfinite(training["volume"])
        groups = training.loc[finite].groupby("minute_of_session", sort=True)["volume"]
        median = groups.median()
        mad = groups.agg(lambda values: (values - values.median()).abs().median())
        scale = 1.4826 * mad
        fallback = median.abs().clip(lower=1.0)
        scale = scale.where(scale > 0, fallback)
        self.statistics_ = pd.DataFrame({"median": median, "mad": mad, "scale": scale})
        return self

    def transform(self, causal: pd.DataFrame) -> pd.Series:
        if self.statistics_ is None:
            raise RuntimeError("VolumeSeasonality must be fitted before transform")
        slots = causal["minute_of_session"]
        center = slots.map(self.statistics_["median"])
        scale = slots.map(self.statistics_["scale"])
        volume = causal["volume"].where(np.isfinite(causal["volume"]))
        return ((volume - center) / scale).rename("volume_surprise")
