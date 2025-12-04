# src/baselines.py
import numpy as np
import pandas as pd

def _is_rebalance_day(idx: pd.DatetimeIndex, freq: str, t: int) -> bool:
    "check if it is rebalanced day; freq ∈ {'D','W','M','Q'}"""
    if freq == 'D':
        return True
    if freq == 'W':
        # every Monday
        return idx[t].weekday() == 0
    if freq == 'M':
        if t == 0: 
            return True
        prev, cur = idx[t-1], idx[t]
        return (prev.month != cur.month)
    if freq == 'Q':
        if t == 0:
            return True
        prev, cur = idx[t-1], idx[t]
        return ( (prev.quarter != cur.quarter) or (prev.year != cur.year) )

    return False

def buy_and_hold(returns: pd.DataFrame, cost_bps: float = 0.0):
    """
    buy and hold (using log return)
    No rebalanced applied. 
    """
    idx = returns.index
    R = returns.values              # log returns [T,N]
    growth = np.exp(R)              
    N = returns.shape[1]
    fee = cost_bps / 1e4

    w = np.ones(N) / N              
    out = []

    for t in range(len(returns)):
        port_growth = float(np.dot(w, growth[t]))
        r_port = np.log(port_growth)

        # For intial transaction fee to enter the market
        if t == 0 and fee > 0:
            turnover = np.abs(w).sum()  
            r_port -= fee * turnover

        out.append(r_port)

        w = (w * growth[t]) / (port_growth + 1e-12)

    ser = pd.Series(out, index=idx)
    return ser, ser.cumsum()

def equal_weight(returns: pd.DataFrame, freq: str = 'M', cost_bps: float = 0.0):
    """
    - balance to equal weights when rebalanced
    - cost = cost_bps/1e4 * ||w_new - w_prev||_1
    """
    idx = returns.index
    R = returns.values
    growth = np.exp(R)
    N = returns.shape[1]
    fee = cost_bps / 1e4

    w = np.ones(N) / N
    out = []

    for t in range(len(returns)):
        port_growth = float(np.dot(w, growth[t]))
        r_port = np.log(port_growth)

        # Check if it is rebalance day
        # initial transaction fee for entering the market is applied
        if _is_rebalance_day(idx, freq, t):
            w_new = np.ones(N) / N
            turnover = np.abs(w_new - w).sum()
            r_port -= fee * turnover
            w = w_new
        else:
            # if not rebalance day, let it drift with the market
            w = (w * growth[t]) / (port_growth + 1e-12)
        
        out.append(r_port)

    ser = pd.Series(out, index=idx)
    return ser, ser.cumsum()
