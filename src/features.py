# src/features.py
import numpy as np
import pandas as pd

def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """把价格宽表转成对数收益宽表"""
    return np.log(prices / prices.shift(1)).dropna()

def build_features(returns: pd.DataFrame, win: int = 20) -> pd.DataFrame:
    """
    最小可用特征：当前各资产log return。
    先跑通闭环，后面再加滚动均值、波动率、RSI等。
    """
    return returns.copy()
