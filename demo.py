# demo.py
from stable_baselines3 import PPO
import pandas as pd

from src.data_loader import load_split_freq
from src.features import to_log_returns, build_features
from src.envs import PortfolioEnv
from src.baselines import buy_and_hold, equal_weight
from src.evaluate import ann_metrics, plot_equity

WINDOW = 20
COST_BPS = 20
TIMESTEPS = 200_000  # 先跑个小步数，确认流程

def run_train_test_daily():
    # 1) 读价格（train/test, daily）
    train_prices = load_split_freq(split="train", freq="daily")
    test_prices  = load_split_freq(split="test",  freq="daily")

    # 2) 转对数收益 + 构建最小特征
    train_r, test_r = to_log_returns(train_prices), to_log_returns(test_prices)
    train_f, test_f = build_features(train_r, WINDOW), build_features(test_r, WINDOW)

    # 3) 训练 PPO（用train）
    env = PortfolioEnv(train_r.loc[train_f.index], train_f, WINDOW, COST_BPS)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=TIMESTEPS)

    # 4) 测试回测（用test）
    test_env = PortfolioEnv(test_r.loc[test_f.index], test_f, WINDOW, COST_BPS)
    obs, _ = test_env.reset()
    rets, done = [], False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, _, _ = test_env.step(action)
        rets.append(r)
    idx = test_f.index[WINDOW:]          # 修复长度不匹配
    rl_ret = pd.Series(rets, index=idx)
    rl_cum = rl_ret.cumsum()
    # baselines 也用同一时段对齐
    test_slice = test_r.loc[test_f.index].iloc[WINDOW:]
    ew_r, ew_cum = equal_weight(test_slice, freq='M', cost_bps=20)
    bh_r, bh_cum = buy_and_hold(test_slice)

    # 5) 基线
    ew_r, ew_cum = equal_weight(test_r.loc[test_f.index])
    bh_r, bh_cum = buy_and_hold(test_r.loc[test_f.index])

    # 6) 指标与图
    print("Metrics:")
    print("  PPO:", ann_metrics(rl_ret))
    print("  EW :", ann_metrics(ew_r))
    print("  BH :", ann_metrics(bh_r))

    plot_equity(
        {"RL_PPO": rl_cum, "EW": ew_cum, "BH": bh_cum},
        "results/figures/equity_daily.png"
    )
    print("Saved figure -> results/figures/equity_daily.png")

if __name__ == "__main__":
    run_train_test_daily()
