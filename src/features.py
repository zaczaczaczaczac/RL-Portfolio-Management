# src/features.py
import numpy as np
import pandas as pd

def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    convert the closed price to the log price 
    """
    return np.log(prices / prices.shift(1)).dropna()

def build_features(returns: pd.DataFrame, win: int = 20) -> pd.DataFrame:
    """
    building the minimal log return feature
    """
    return returns.copy()
