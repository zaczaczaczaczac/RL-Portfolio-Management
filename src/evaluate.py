# src/evaluate.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

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
