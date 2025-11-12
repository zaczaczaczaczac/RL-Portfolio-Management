# src/baselines.py
import numpy as np
import pandas as pd

def buy_and_hold(returns: pd.DataFrame):
    """
    初始等权，之后不再调仓。返回(逐日组合return, 累计log return)
    """
    N = returns.shape[1]
    w = np.ones(N) / N
    port_ret = (returns.values @ w)  # 各日组合log return
    series = pd.Series(port_ret, index=returns.index)
    return series, series.cumsum()

def equal_weight_daily(returns: pd.DataFrame, cost_bps: float = 0.0):
    """
    每日等权再平衡；如需扣除成本，可设置 cost_bps>0，按简单的日常换手估计。
    """
    N = returns.shape[1]
    w_prev = np.ones(N) / N
    cost = cost_bps / 1e4
    out = []

    for t in range(len(returns)):
        r = float((returns.iloc[t].values * w_prev).sum())
        # 每日重回等权，换手率 = |w_new - w_prev|_1
        w_new = np.ones(N) / N
        turnover = np.abs(w_new - w_prev).sum()
        r_net = r - cost * turnover
        out.append(r_net)
        w_prev = w_new

    ser = pd.Series(out, index=returns.index)
    return ser, ser.cumsum()
