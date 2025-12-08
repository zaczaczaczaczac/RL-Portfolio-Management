# src/features.py
import numpy as np
import pandas as pd


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute 1-step log returns from price DataFrame."""
    return np.log(prices / prices.shift(1)).dropna()


def build_features(
    returns: pd.DataFrame,
    win: int = 20,
    config: str = "F0",
) -> pd.DataFrame:
    """
    Build feature matrix from per-asset log returns.

    Feature configs:
      F0: baseline, 1-step returns only
      F1: returns + rolling mean + rolling volatility
      F2: F1 + Sharpe-like ratio + rolling drawdown
      F3: F2 + simple technical indicators (MA diff, RSI)

    Parameters
    ----------
    returns : DataFrame
        Log returns, columns = assets.
    win : int
        Base window size (used as default for rolling stats).
    config : {"F0","F1","F2","F3"}
        Which feature set to construct.

    Returns
    -------
    DataFrame
        Feature matrix with all assets stacked along columns.
    """
    r = returns.copy().astype(float)
    feats = []

    # ---- F0: always include 1-step returns ----
    base = r.add_prefix("ret_")
    feats.append(base)

    # short/long horizons relative to win
    short_win = max(2, win // 4)   # e.g. 5 if win=20
    long_win = win                 # e.g. 20
    vol_win = win
    sharpe_win = win
    dd_win = win * 3               # longer window for drawdown

    # helper for RSI on returns
    def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
        """
        RSI that always returns a Series with the SAME index/length
        as `series` (NaNs at the beginning are fine – we drop them later).
        """
        delta = series.diff()

        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)

        roll_up = up.rolling(window, min_periods=1).mean()
        roll_down = down.rolling(window, min_periods=1).mean()

        rs = roll_up / (roll_down + 1e-8)
        rsi = 100.0 - 100.0 / (1.0 + rs)

        # no dropna here – keep full length, we’ll drop jointly at the end
        return rsi

    # ---- F1: add rolling mean & volatility ----
    if config in ("F1", "F2", "F3"):
        mean_short = r.rolling(short_win, min_periods=1).mean().add_prefix(
            f"mean{short_win}_"
        )
        mean_long = r.rolling(long_win, min_periods=1).mean().add_prefix(
            f"mean{long_win}_"
        )
        vol = r.rolling(vol_win, min_periods=1).std().add_prefix(
            f"vol{vol_win}_"
        )

        feats.extend([mean_short, mean_long, vol])

    # ---- F2: add Sharpe-like ratio + drawdown ----
    if config in ("F2", "F3"):
        roll_mean = r.rolling(sharpe_win, min_periods=1).mean()
        roll_std = r.rolling(sharpe_win, min_periods=1).std()
        sharpe_like = (roll_mean / (roll_std + 1e-8)).add_prefix(
            f"sharpe{sharpe_win}_"
        )

        # rolling drawdown based on cumulative log returns
        cum_log = r.cumsum()
        pseudo_price = np.exp(cum_log)
        roll_max = pseudo_price.rolling(dd_win, min_periods=1).max()
        drawdown = (pseudo_price / (roll_max + 1e-8) - 1.0).add_prefix(
            f"dd{dd_win}_"
        )

        feats.extend([sharpe_like, drawdown])

    # ---- F3: add simple technical indicators on returns ----
    if config == "F3":
        # moving-average "trend strength" on returns
        ma_short = r.rolling(short_win, min_periods=1).mean()
        ma_long = r.rolling(long_win * 2, min_periods=1).mean()
        ma_diff = (ma_short - ma_long).add_prefix(
            f"maDiff{short_win}_{long_win*2}_"
        )

        # RSI on returns
        rsi_all = pd.DataFrame(
            {col: _rsi(r[col], window=14) for col in r.columns},
            index=r.index,
        ).add_prefix("rsi14_")

        feats.extend([ma_diff, rsi_all])

    feat_df = pd.concat(feats, axis=1)
    feat_df = feat_df.dropna()

    return feat_df
