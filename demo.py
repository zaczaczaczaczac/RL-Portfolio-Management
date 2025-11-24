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

def run_train_test_daily_raw():
    # 1) 读价格（train/test, daily）
    train_prices = load_split_freq(split="train", freq="daily")
    test_prices  = load_split_freq(split="test",  freq="daily")

    # 2) 转对数收益 + 构建最小特征
    train_r, test_r = to_log_returns(train_prices), to_log_returns(test_prices)
    train_f, test_f = build_features(train_r, WINDOW), build_features(test_r, WINDOW)

    # 3) 训练 PPO（raw reward）
    env = PortfolioEnv(
        train_r.loc[train_f.index],
        train_f,
        window=WINDOW,
        cost_bps=COST_BPS,
        reward_mode="raw",      # ✅ 明确写 raw
        lambda_risk=0.0,
    )
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=TIMESTEPS)

    # 4) 测试回测（raw reward）
    test_env = PortfolioEnv(
        test_r.loc[test_f.index],
        test_f,
        window=WINDOW,
        cost_bps=COST_BPS,
        reward_mode="raw",
        lambda_risk=0.0,
    )
    obs, _ = test_env.reset()
    rets, done = [], False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, _, _ = test_env.step(action)
        rets.append(r)

    idx = test_f.index[WINDOW:]          # 对齐 reward 长度
    rl_ret = pd.Series(rets, index=idx)
    rl_cum = rl_ret.cumsum()

    # baselines 也用同一时段对齐
    test_slice = test_r.loc[test_f.index].iloc[WINDOW:]
    ew_r, ew_cum = equal_weight(test_slice, freq='M', cost_bps=20)
    bh_r, bh_cum = buy_and_hold(test_slice)

    print("Metrics (RAW reward):")
    print("  PPO:", ann_metrics(rl_ret))
    print("  EW :", ann_metrics(ew_r))
    print("  BH :", ann_metrics(bh_r))

    plot_equity(
        {"RL_PPO_raw": rl_cum, "EW": ew_cum, "BH": bh_cum},
        "results/figures/equity_daily_raw.png"
    )
    print("Saved figure -> results/figures/equity_daily_raw.png")

def run_train_test_daily_risk():
    # 1) 读价格（与 raw 相同）
    train_prices = load_split_freq(split="train", freq="daily")
    test_prices  = load_split_freq(split="test",  freq="daily")

    train_r, test_r = to_log_returns(train_prices), to_log_returns(test_prices)
    train_f, test_f = build_features(train_r, WINDOW), build_features(test_r, WINDOW)

    # 2) 训练 PPO（risk-adjusted reward）
    env = PortfolioEnv(
        train_r.loc[train_f.index],
        train_f,
        window=WINDOW,
        cost_bps=COST_BPS,
        reward_mode="risk",     # ✅ 使用 risk 模式
        lambda_risk=5.0,        # 可以之后调参数，比如 2.0, 5.0, 10.0 做消融
        vol_window=20,
    )
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=TIMESTEPS)

    # 3) 测试回测（risk 模式）
    test_env = PortfolioEnv(
        test_r.loc[test_f.index],
        test_f,
        window=WINDOW,
        cost_bps=COST_BPS,
        reward_mode="risk",
        lambda_risk=5.0,
        vol_window=20,
    )
    obs, _ = test_env.reset()
    rets, done = [], False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, _, _ = test_env.step(action)
        rets.append(r)

    idx = test_f.index[WINDOW:]
    rl_ret = pd.Series(rets, index=idx)
    rl_cum = rl_ret.cumsum()

    # baselines 同样区间
    test_slice = test_r.loc[test_f.index].iloc[WINDOW:]
    ew_r, ew_cum = equal_weight(test_slice, freq='M', cost_bps=20)
    bh_r, bh_cum = buy_and_hold(test_slice)

    print("Metrics (RISK-ADJUSTED reward):")
    print("  PPO_risk:", ann_metrics(rl_ret))
    print("  EW      :", ann_metrics(ew_r))
    print("  BH      :", ann_metrics(bh_r))

    plot_equity(
        {"RL_PPO_risk": rl_cum, "EW": ew_cum, "BH": bh_cum},
        "results/figures/equity_daily_risk.png"
    )
    print("Saved figure -> results/figures/equity_daily_risk.png")

if __name__ == "__main__":
    run_train_test_daily_raw()
    run_train_test_daily_risk()
