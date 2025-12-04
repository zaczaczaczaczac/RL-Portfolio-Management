# src/data_loader.py
import os
import glob
import pandas as pd

PRICE_COL_ORDER = ["Adj Close", "Adj_Close", "AdjClose", "Close", "close"]
DATE_COL_CANDIDATES = ["Date", "Datetime", "date", "datetime", "timestamp"]

def _detect_date_col(df: pd.DataFrame) -> str:
    for c in DATE_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError("No date/datetime column found. Tried: " + ", ".join(DATE_COL_CANDIDATES))

def _detect_price_col(df: pd.DataFrame) -> str:
    for c in PRICE_COL_ORDER:
        if c in df.columns:
            return c
    
    non_date_cols = [c for c in df.columns if c not in DATE_COL_CANDIDATES]
    if len(non_date_cols) == 1:
        return non_date_cols[0]
    raise ValueError(f"Cannot determine price column from {df.columns.tolist()}")

def _ticker_from_filename(path: str) -> str:
    base = os.path.basename(path)
    return base.split("_")[0].upper()

def _strip_tz(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """convert from UTC timezone to naive (UTC→naive)"""
    if getattr(idx, "tz", None) is not None:
        return idx.tz_convert("UTC").tz_localize(None)
    return idx

def _read_one_csv(path: str) -> pd.Series:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    dcol = _detect_date_col(df)
    pcol = _detect_price_col(df)

    df[dcol] = pd.to_datetime(df[dcol], errors="coerce", utc=True)
    df = df.dropna(subset=[dcol]).sort_values(dcol).set_index(dcol)

    s = df[pcol].astype("float64")
    s.index = _strip_tz(s.index)              
    s = s[~s.index.duplicated(keep="first")]  
    s.name = _ticker_from_filename(path)
    return s

def load_folder_to_prices(folder: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV in: {folder}")
    series_list = [_read_one_csv(f) for f in files]
    prices = pd.concat(series_list, axis=1).dropna(how="any").sort_index()
    prices = prices.reindex(sorted(prices.columns), axis=1)  # fix temporal series ordering
    return prices

def load_split_freq(root="data", split="train", freq="daily") -> pd.DataFrame:
    folder = os.path.join(root, split, freq)
    return load_folder_to_prices(folder)
