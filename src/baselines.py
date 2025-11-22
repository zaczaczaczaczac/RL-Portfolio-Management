# src/baselines.py
import numpy as np
import pandas as pd

def _is_rebalance_day(idx: pd.DatetimeIndex, freq: str, t: int) -> bool:
    """根据频率判断是否再平衡日。freq ∈ {'D','W','M','Q'}"""
    if freq == 'D':
        return True
    if freq == 'W':
        # 每周一（或你喜欢的周频规则）
        return idx[t].weekday() == 0
    if freq == 'M':
        # 月末/月初再平衡都行，这里用“月初第一天有交易日”：
        if t == 0: 
            return True
        prev, cur = idx[t-1], idx[t]
        return (prev.month != cur.month)
    if freq == 'Q':
        if t == 0:
            return True
        prev, cur = idx[t-1], idx[t]
        return ( (prev.quarter != cur.quarter) or (prev.year != cur.year) )
    # 默认不再平衡
    return False

def buy_and_hold(returns: pd.DataFrame, cost_bps: float = 0.0):
    """
    真正的 Buy-and-Hold（用 log return 严谨实现）：
    - t=0 等权
    - 每天用“前一日权重”计算组合回报
    - 价格变动后，权重根据各资产增长自然漂移
    """
    idx = returns.index
    R = returns.values              # log returns [T,N]
    growth = np.exp(R)              # 每日增长因子
    N = returns.shape[1]
    fee = cost_bps / 1e-4

    w = np.ones(N) / N              # 初始等权
    out = []

    for t in range(len(returns)):
        # 组合当天增长因子 = 权重加权资产增长因子
        port_growth = float(np.dot(w, growth[t]))
        # 当日组合 log 回报
        r_port = np.log(port_growth)

        # For intial transaction fee to enter the market
        if t == 0 and fee > 0:
            turnover = np.abs(w).sum()  
            r_port -= fee * turnover

        out.append(r_port)

        # 用当日增长后更新“自然漂移”的新权重（不再平衡）
        w = (w * growth[t]) / (port_growth + 1e-12)

    ser = pd.Series(out, index=idx)
    return ser, ser.cumsum()

def equal_weight(returns: pd.DataFrame, freq: str = 'M', cost_bps: float = 0.0):
    """
    等权策略（可设再平衡频率：'D'/'W'/'M'/'Q'），含简单的 L1 换手成本。
    - 每到再平衡日把权重重置为等权
    - 成本 = cost_bps/1e4 * ||w_new - w_prev||_1
    """
    idx = returns.index
    R = returns.values
    growth = np.exp(R)
    N = returns.shape[1]
    fee = cost_bps / 1e4

    # 初始权重
    w = np.ones(N) / N
    out = []

    for t in range(len(returns)):
        # 用上一时刻权重计算当日组合回报
        port_growth = float(np.dot(w, growth[t]))
        r_port = np.log(port_growth)

        # 是否再平衡
        # initial transaction fee for entering the market is applied
        if _is_rebalance_day(idx, freq, t):
            w_new = np.ones(N) / N
            turnover = np.abs(w_new - w).sum()
            r_port -= fee * turnover
            w = w_new
        else:
            # 非再平衡日，权重自然漂移（以便下一次再平衡时计算换手）
            w = (w * growth[t]) / (port_growth + 1e-12)
        
        out.append(r_port)

    ser = pd.Series(out, index=idx)
    return ser, ser.cumsum()
