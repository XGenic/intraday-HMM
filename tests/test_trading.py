import numpy as np
import pandas as pd
import pytest

from regime_retrieval.config import CostsConfig, ResearchConfig, TradingConfig
from regime_retrieval.evaluation.trading import simulate_trading


def prepared_bars(closes, opens=None, timestamps=None):
    if timestamps is None:
        timestamps = pd.date_range("2025-01-02T14:35Z", periods=len(closes), freq="5min")
    timestamps = pd.DatetimeIndex(timestamps)
    sessions = timestamps.tz_localize(None).normalize()
    boundaries = pd.Series(sessions).ne(pd.Series(sessions).shift()) | pd.Series(
        timestamps
    ).diff().ne(pd.Timedelta(minutes=5))
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "session": sessions,
            "segment_id": boundaries.cumsum().to_numpy(),
            "open": closes if opens is None else opens,
            "close": closes,
        }
    )


def predictions(bars, windows, medians=None, method="raw_knn", horizon=5):
    return pd.DataFrame(
        {
            "method": method,
            "horizon_minutes": horizon,
            "query_timestamp": [bars.iloc[q]["timestamp"] for q, _ in windows],
            "actual_outcome_end": [bars.iloc[x]["timestamp"] for _, x in windows],
            "median": [0.01] * len(windows) if medians is None else medians,
        }
    )


def config(bps=0.0, threshold=0.0):
    return ResearchConfig(
        costs=CostsConfig(bps_per_side=bps),
        trading=TradingConfig(enabled=True, threshold=threshold),
    )


def test_signal_enters_next_open_not_query_open_or_query_close():
    bars = prepared_bars([20, 110, 500], opens=[10, 100, 1000])
    metrics, trades, equity = simulate_trading(predictions(bars, [(0, 1)]), bars, config())
    trade = trades.iloc[0]
    assert trade["entry_price"] == 100
    assert trade["exit_price"] == 110
    assert trade["entry_timestamp"] == bars.iloc[0]["timestamp"]
    assert trade["entry_bar_end"] == bars.iloc[1]["timestamp"]
    assert trade["exit_timestamp"] == bars.iloc[1]["timestamp"]
    assert trade["gross_return"] == pytest.approx(0.1)
    assert metrics.iloc[0]["net_return"] == pytest.approx(0.1)
    assert equity["net_equity"].tolist() == pytest.approx([1, 1.1, 1.1])


def test_overlapping_signals_suppressed_but_exit_boundary_can_reenter():
    bars = prepared_bars([100, 100, 110, 110, 121], opens=[100, 100, 100, 110, 110])
    frame = predictions(bars, [(0, 2), (1, 3), (2, 4)], horizon=10)
    metrics, trades, _ = simulate_trading(frame.iloc[::-1], bars, config())
    assert trades["query_timestamp"].tolist() == bars.iloc[[0, 2]]["timestamp"].tolist()
    assert metrics.iloc[0]["n_trades"] == 2
    assert metrics.iloc[0]["gross_return"] == pytest.approx(0.21)
    assert metrics.iloc[0]["time_in_market"] == pytest.approx(4 / 5)


@pytest.mark.parametrize(
    "timestamps,window",
    [
        (["2025-01-02T14:35Z", "2025-01-02T14:45Z", "2025-01-02T14:50Z"], (0, 2)),
        (["2025-01-02T14:35Z", "2025-01-02T14:40Z", "2025-01-02T14:50Z"], (0, 2)),
        (["2025-01-02T20:55Z", "2025-01-02T21:00Z", "2025-01-03T14:35Z"], (1, 2)),
        (["2025-01-02T14:35Z", "2025-01-02T14:40Z", "2025-01-02T14:45Z"], (2, 2)),
    ],
)
def test_no_entry_or_holding_across_gap_session_or_missing_next_bar(timestamps, window):
    bars = prepared_bars([100, 110, 120], timestamps=timestamps)
    # Even stale segment metadata must not permit crossing an actual gap/session.
    bars["segment_id"] = 0
    metrics, trades, equity = simulate_trading(predictions(bars, [window]), bars, config())
    assert trades.empty
    assert metrics.iloc[0]["n_trades"] == 0
    assert equity["net_equity"].eq(1).all()


def test_segment_boundary_blocks_trade_even_with_contiguous_timestamps():
    bars = prepared_bars([100, 110, 120])
    bars["segment_id"] = [0, 0, 1]
    _, trades, _ = simulate_trading(predictions(bars, [(0, 2)]), bars, config())
    assert trades.empty


def test_costs_use_self_financing_units_on_both_sides_and_compound():
    bars = prepared_bars([100, 110, 110, 120], opens=[100, 100, 110, 100])
    metrics, trades, equity = simulate_trading(
        predictions(bars, [(0, 1), (2, 3)]), bars, config(bps=100)
    )
    first_units = 1 / 101
    first_after = first_units * 110 * 0.99
    second_units = first_after / 101
    second_after = second_units * 120 * 0.99
    assert trades["net_units"].tolist() == pytest.approx([first_units, second_units])
    assert trades["entry_cost"].tolist() == pytest.approx([first_units, second_units])
    assert trades["exit_cost"].tolist() == pytest.approx([first_units * 1.1, second_units * 1.2])
    assert trades["net_return"].tolist() == pytest.approx(
        [1.1 * 0.99 / 1.01 - 1, 1.2 * 0.99 / 1.01 - 1]
    )
    assert equity["net_equity"].tolist() == pytest.approx(
        [1 / 1.01, first_after, first_after / 1.01, second_after]
    )
    assert metrics.iloc[0]["gross_return"] == pytest.approx(0.32)
    assert metrics.iloc[0]["net_return"] == pytest.approx(second_after - 1)
    assert metrics.iloc[0]["turnover"] == pytest.approx(first_units * 210 + second_units * 220)
    assert trades["costs"].sum() == pytest.approx(metrics.iloc[0]["turnover"] * 0.01)


def test_mark_to_market_drawdown_includes_loss_before_profitable_liquidation():
    bars = prepared_bars([100, 120, 80, 110], opens=[100, 100, 120, 80])
    metrics, trades, equity = simulate_trading(
        predictions(bars, [(0, 3)], horizon=15), bars, config()
    )
    assert trades.iloc[0]["net_return"] == pytest.approx(0.1)
    assert equity["net_equity"].tolist() == pytest.approx([1, 1.2, 0.8, 1.1])
    assert metrics.iloc[0]["max_drawdown"] == pytest.approx(1 - 0.8 / 1.2)
    assert np.isnan(metrics.iloc[0]["sharpe"])


def test_entry_fee_drawdown_visible_even_when_first_bar_gain_covers_cost():
    bars = prepared_bars([100, 110], opens=[100, 100])
    metrics, _, equity = simulate_trading(predictions(bars, [(0, 1)]), bars, config(bps=100))
    assert equity.iloc[0]["net_equity"] == pytest.approx(1 / 1.01)
    assert metrics.iloc[0]["max_drawdown"] == pytest.approx(1 - 1 / 1.01)


def test_daily_sharpe_includes_no_trade_session_but_excludes_training_and_later_sessions():
    timestamps = [f"2025-01-{day:02}T{time}Z" for day in range(1, 6) for time in ["14:35", "14:40"]]
    bars = prepared_bars(
        [100, 100, 100, 110, 100, 100, 100, 100, 100, 100], opens=[100] * 10, timestamps=timestamps
    )
    frame = predictions(bars, [(2, 3), (6, 7)], medians=[0.01, -0.01])
    metrics, trades, equity = simulate_trading(frame, bars, config())
    daily = np.array([0.1, 0, 0])
    assert metrics.iloc[0]["sharpe"] == pytest.approx(
        np.sqrt(252) * daily.mean() / daily.std(ddof=1)
    )
    assert metrics.iloc[0]["n_sessions"] == 3
    assert metrics.iloc[0]["n_bars"] == 6
    assert metrics.iloc[0]["time_in_market"] == pytest.approx(1 / 6)
    assert equity["timestamp"].tolist() == bars.iloc[2:8]["timestamp"].tolist()
    assert len(trades) == 1


def test_threshold_is_strict_long_only_and_no_trade_metrics_are_explicit():
    timestamps = [
        "2025-01-02T14:35Z",
        "2025-01-02T14:40Z",
        "2025-01-03T14:35Z",
        "2025-01-03T14:40Z",
    ]
    bars = prepared_bars([100, 90, 100, 90], timestamps=timestamps)
    metrics, trades, equity = simulate_trading(
        predictions(bars, [(0, 1), (2, 3)], medians=[0.01, -0.2]),
        bars,
        config(threshold=0.01),
    )
    result = metrics.iloc[0]
    assert trades.empty
    assert {"entry_timestamp", "exit_timestamp", "net_return", "costs"} <= set(trades.columns)
    assert result["n_trades"] == 0
    assert result["net_return"] == result["gross_return"] == 0
    assert result["max_drawdown"] == result["time_in_market"] == result["turnover"] == 0
    assert np.isnan(result["sharpe"])
    assert np.isnan(result["hit_rate"])
    assert np.isnan(result["average_holding_return"])
    assert equity["gross_equity"].eq(1).all()
    assert equity["net_equity"].eq(1).all()


def test_methods_and_horizons_have_independent_accounts_and_positions():
    bars = prepared_bars([100, 110, 120], opens=[100, 100, 100])
    frame = pd.concat(
        [
            predictions(bars, [(0, 1)], method="raw_knn", horizon=5),
            predictions(bars, [(0, 2)], method="raw_knn", horizon=10),
            predictions(bars, [(0, 1)], method="hmm_trajectory", horizon=5),
        ],
        ignore_index=True,
    )
    metrics, trades, _ = simulate_trading(frame, bars, config())
    returns = metrics.set_index(["method", "horizon_minutes"])["net_return"]
    assert returns["raw_knn", 5] == pytest.approx(0.1)
    assert returns["raw_knn", 10] == pytest.approx(0.2)
    assert returns["hmm_trajectory", 5] == pytest.approx(0.1)
    assert len(trades) == 3


def test_empty_predictions_return_stable_empty_tables():
    bars = prepared_bars([100, 110])
    frames = simulate_trading(pd.DataFrame(), bars, config())
    assert all(frame.empty for frame in frames)
    assert {"n_trades", "sharpe", "max_drawdown"} <= set(frames[0].columns)
    assert {"entry_timestamp", "exit_timestamp", "costs"} <= set(frames[1].columns)
    assert {"timestamp", "session", "gross_equity", "net_equity"} <= set(frames[2].columns)
