"""Secondary long/flat simulation with observable execution and session-level risk."""

import numpy as np
import pandas as pd

from regime_retrieval.config import ResearchConfig

METRIC_COLUMNS = [
    "method",
    "horizon_minutes",
    "n_trades",
    "n_sessions",
    "n_bars",
    "gross_return",
    "net_return",
    "sharpe",
    "max_drawdown",
    "hit_rate",
    "average_holding_return",
    "turnover",
    "time_in_market",
    "bps_per_side",
]
TRADE_COLUMNS = [
    "method",
    "horizon_minutes",
    "query_timestamp",
    "session",
    "entry_timestamp",
    "entry_bar_end",
    "exit_timestamp",
    "entry_price",
    "exit_price",
    "gross_return",
    "net_return",
    "net_units",
    "entry_cost",
    "exit_cost",
    "costs",
    "entry_notional",
    "exit_notional",
    "net_equity_before",
    "net_equity_after",
    "holding_bars",
]
EQUITY_COLUMNS = [
    "method",
    "horizon_minutes",
    "timestamp",
    "session",
    "gross_equity",
    "net_equity",
]


def simulate_trading(
    predictions: pd.DataFrame, bars: pd.DataFrame, config: ResearchConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate each method/horizon separately, starting each account at one unit.

    Enter long only when the predicted median exceeds the fixed threshold. A
    signal is available at its query bar END; execution is at the NEXT contiguous
    bar OPEN. Those timestamps coincide as an idealized boundary execution with
    no additional latency. The close at ``actual_outcome_end`` liquidates the
    position. Query, entry, and exit must belong to one contiguous session segment;
    invalid execution windows are skipped. Queries strictly before the previous
    exit are suppressed; a query at that exit may immediately open the next trade.

    With equity E, open price P, close price X, and per-side cost c=bps/10000,
    buy u=E/[P*(1+c)] fractional units, paying c*u*P on entry and c*u*X on exit.
    Final equity is u*X*(1-c), so holding net return is
    (X/P)*(1-c)/(1+c)-1. Gross equity is a separate cost-free account following
    the identical trades. No leverage, short positions, or overnight holdings.

    Equity marks every observed prepared bar close. At an entry boundary, the
    query timestamp instead records post-entry equity marked at the next open
    (including its fee), after any prior liquidation at that boundary. Drawdown
    uses these net marks and the initial capital, not just closed-trade returns.

    Every track covers whole observed sessions from the earliest through latest
    query session, including intermediate sessions without predictions or trades;
    training sessions and sessions after that span are excluded. Daily net returns
    use each session's last mark and an initial equity of one. Sharpe is their mean
    divided by sample standard deviation, annualized by sqrt(252); fewer than two
    sessions or zero variance gives NaN. Missing whole sessions are not fabricated.

    n_trades counts executed round trips; hit_rate and average_holding_return use
    their net holding returns (NaN with no trades). Turnover is summed entry plus
    exit notional in the net account divided by initial capital (one), without
    annualization or a session denominator. time_in_market is held bar intervals
    divided by all observed bar intervals in the span (n_bars); n_sessions exposes
    the daily-return denominator. Empty input returns three stable empty schemas.
    """
    if predictions.empty:
        return (
            pd.DataFrame(columns=METRIC_COLUMNS),
            pd.DataFrame(columns=TRADE_COLUMNS),
            pd.DataFrame(columns=EQUITY_COLUMNS),
        )

    timestamps = pd.DatetimeIndex(bars["timestamp"])
    if not timestamps.is_unique or not timestamps.is_monotonic_increasing:
        raise ValueError("Trading requires unique, chronologically ordered prepared bars")
    prices = bars[["open", "close"]].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Trading requires finite positive open and close prices")
    if predictions.duplicated(["method", "horizon_minutes", "query_timestamp"]).any():
        raise ValueError("Trading requires one prediction per method, horizon, and query")
    query_positions = timestamps.get_indexer(predictions["query_timestamp"])
    if (query_positions < 0).any():
        raise ValueError("Trading query timestamps must exist in prepared bars")

    interval = pd.Timedelta(minutes=config.bar_minutes)
    # Check actual intervals as well as metadata: missing bars must never be bridged.
    boundaries = (
        bars["session"].ne(bars["session"].shift())
        | bars["segment_id"].ne(bars["segment_id"].shift())
        | bars["timestamp"].diff().ne(interval)
    ).to_numpy()
    segments = boundaries.cumsum()
    sessions = bars["session"].to_numpy()
    first_session = sessions[query_positions.min()]
    last_session = sessions[query_positions.max()]
    observed = np.flatnonzero((sessions >= first_session) & (sessions <= last_session))
    first, stop = int(observed[0]), int(observed[-1]) + 1
    span = bars.iloc[first:stop]
    n_bars = len(span)
    n_sessions = int(span["session"].nunique())
    cost = config.costs.bps_per_side / 10000
    metrics = []
    trades = []
    equities = []

    for (method, horizon), track in predictions.groupby(["method", "horizon_minutes"], sort=True):
        gross_equity = np.ones(n_bars)
        net_equity = np.ones(n_bars)
        gross_cash = net_cash = 1.0
        cursor = 0
        last_exit = -1
        holding_bars = 0
        track_returns = []
        turnover = 0.0
        signals = track.loc[track["median"].gt(config.trading.threshold)].sort_values(
            "query_timestamp"
        )
        query_indices = timestamps.get_indexer(signals["query_timestamp"])
        exit_indices = timestamps.get_indexer(signals["actual_outcome_end"])
        for query, exit_index in zip(query_indices, exit_indices, strict=True):
            entry = query + 1
            if (
                query < last_exit
                or entry >= len(bars)
                or exit_index < entry
                or exit_index >= stop
                or segments[query] != segments[exit_index]
            ):
                continue
            entry_price, exit_price = prices[entry, 0], prices[exit_index, 1]
            q, e, x = query - first, entry - first, exit_index - first
            gross_equity[cursor : q + 1] = gross_cash
            net_equity[cursor : q + 1] = net_cash
            net_units = net_cash / (entry_price * (1 + cost))
            entry_notional = net_units * entry_price
            exit_notional = net_units * exit_price
            entry_cost, exit_cost = cost * entry_notional, cost * exit_notional
            gross_return = exit_price / entry_price - 1
            net_return = (exit_price / entry_price) * (1 - cost) / (1 + cost) - 1
            # At the query/next-open boundary the only immediate equity loss is fee.
            net_equity[q] = entry_notional
            closes = prices[entry : exit_index + 1, 1]
            gross_equity[e : x + 1] = (gross_cash / entry_price) * closes
            net_equity[e : x + 1] = net_units * closes
            gross_cash = float(gross_equity[x])
            net_after = exit_notional - exit_cost
            net_equity[x] = net_after
            held = exit_index - query
            trades.append(
                {
                    "method": method,
                    "horizon_minutes": horizon,
                    "query_timestamp": timestamps[query],
                    "session": sessions[query],
                    "entry_timestamp": timestamps[entry] - interval,
                    "entry_bar_end": timestamps[entry],
                    "exit_timestamp": timestamps[exit_index],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_return": gross_return,
                    "net_return": net_return,
                    "net_units": net_units,
                    "entry_cost": entry_cost,
                    "exit_cost": exit_cost,
                    "costs": entry_cost + exit_cost,
                    "entry_notional": entry_notional,
                    "exit_notional": exit_notional,
                    "net_equity_before": net_cash,
                    "net_equity_after": net_after,
                    "holding_bars": held,
                }
            )
            net_cash = net_after
            cursor = x + 1
            last_exit = exit_index
            holding_bars += held
            track_returns.append(net_return)
            turnover += entry_notional + exit_notional

        gross_equity[cursor:] = gross_cash
        net_equity[cursor:] = net_cash
        marks = span[["timestamp", "session"]].reset_index(drop=True).copy()
        marks["method"] = method
        marks["horizon_minutes"] = horizon
        marks["gross_equity"] = gross_equity
        marks["net_equity"] = net_equity
        equities.append(marks[EQUITY_COLUMNS])
        daily_closes = marks.groupby("session", sort=False)["net_equity"].last().to_numpy()
        daily_returns = daily_closes / np.r_[1.0, daily_closes[:-1]] - 1
        daily_std = float(np.std(daily_returns, ddof=1)) if n_sessions > 1 else np.nan
        sharpe = (
            float(np.sqrt(252) * daily_returns.mean() / daily_std)
            if np.isfinite(daily_std) and daily_std > 0
            else np.nan
        )
        peaks = np.maximum.accumulate(np.maximum(net_equity, 1.0))
        metrics.append(
            {
                "method": method,
                "horizon_minutes": horizon,
                "n_trades": len(track_returns),
                "n_sessions": n_sessions,
                "n_bars": n_bars,
                "gross_return": gross_cash - 1,
                "net_return": net_cash - 1,
                "sharpe": sharpe,
                "max_drawdown": float(np.max(1 - net_equity / peaks)),
                "hit_rate": float(np.mean(np.asarray(track_returns) > 0))
                if track_returns
                else np.nan,
                "average_holding_return": float(np.mean(track_returns))
                if track_returns
                else np.nan,
                "turnover": turnover,
                "time_in_market": holding_bars / n_bars,
                "bps_per_side": config.costs.bps_per_side,
            }
        )

    return (
        pd.DataFrame(metrics, columns=METRIC_COLUMNS),
        pd.DataFrame(trades, columns=TRADE_COLUMNS),
        pd.concat(equities, ignore_index=True)[EQUITY_COLUMNS],
    )
