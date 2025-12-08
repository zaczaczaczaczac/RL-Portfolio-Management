# src/evaluate.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def ann_freq_for(freq_str: str) -> int:
    """
    Map data frequency string to number of periods per year
    for annualization in ann_metrics.

    - "daily"  -> 252 trading days
    - "hourly" -> ~6 bars/day * 252
    - "30min"  -> ~13 bars/day * 252
    """
    if freq_str == "daily":
        return 252
    if freq_str == "hourly":
        return 252 * 6
    if freq_str == "30min":
        return 252 * 13
    # fallback
    return 252

def ann_metrics(r: pd.Series, freq=252):
    mu = r.mean() * freq
    vol = r.std() * np.sqrt(freq)
    sharpe = mu / (vol + 1e-12)
    cum = r.cumsum()
    dd = cum - cum.cummax()
    mdd = dd.min()
    return {"AnnRet": mu, "AnnVol": vol, "Sharpe": sharpe, "MaxDD": mdd}

def plot_equity(curves: dict, path_png: str):
    plt.figure()
    for name, ser in curves.items():
        plt.plot(ser.index, ser.values, label=name)
    plt.legend(); plt.title("Cumulative Log Return")
    plt.xlabel("Time"); plt.ylabel("log cum ret")
    plt.tight_layout(); 
    
    dir_name = os.path.dirname(path_png)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    plt.savefig(path_png, dpi=150)
    plt.close()

def save_equity_to_csv(res: dict, path_csv: str):
    # create a new folder if current folder doesn't exist
    dir_name = os.path.dirname(path_csv)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    eq_list = [i for i, _ in res.items()]
    res_list = [r for _, r in res.items()]
    df = pd.DataFrame(data=res_list, index=eq_list)
    df.to_csv(path_csv)
    return None

def save_cum_ret_to_csv(cum_ret_dict: dict, path_csv: str):
    dir_name = os.path.dirname(path_csv)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    equity_list = [i for i, _ in cum_ret_dict.items()]
    ppo = cum_ret_dict["PPO"]
    dqn = cum_ret_dict["DQN"]
    ew = cum_ret_dict["EW"]
    bh = cum_ret_dict["BH"]
    
    df = pd.concat([ppo, dqn, ew, bh], axis=1)
    df.columns = equity_list
    df.to_csv(path_csv)
    return None
