# demo.py

from stable_baselines3 import PPO  # used indirectly via PPO_Agent
import pandas as pd
import torch

from src.data_loader import load_split_freq
from src.features import to_log_returns, build_features
from src.envs import PortfolioEnv, DiscretePortfolioEnv
from src.baselines import buy_and_hold, equal_weight
from src.agents.ppo_agent import PPO_Agent, PPOHyperparams
from src.agents.dqn_agent import DQN_Agent, DQNHyperparams
from src.agents.ppo_lstm_agent import PPO_LSTM_Agent, PPOLSTMHyperparams
from src.evaluate import ann_metrics, plot_equity, save_equity_to_csv, save_cum_ret_to_csv

WINDOW = 20
COST_BPS = 20
TIMESTEPS = 350_000  # adjust if needed


# -------------------- shared helpers -------------------- #

def train_ppo_and_dqn(
    train_r,
    train_f,
    reward_mode: str,
    lambda_risk: float = 0.0,
    vol_window: int = 20,
):
    """
    Given training returns/features and reward settings, construct:
      - PPO agent on PortfolioEnv (continuous actions)
      - DQN agent on DiscretePortfolioEnv (discrete actions)
    Train both and return them.
    """

    # ----- PPO env (continuous actions) -----
    ppo_env = PortfolioEnv(
        train_r.loc[train_f.index],
        train_f,
        window=WINDOW,
        cost_bps=COST_BPS,
        reward_mode=reward_mode,
        lambda_risk=lambda_risk,
        vol_window=vol_window,
    )

    raw_ppo_hyperparams = PPOHyperparams(
        policy="MlpPolicy",
        learning_rate=3e-4,
        n_epochs=12,
        batch_size=32,
        n_steps=2048,
        gae_lambda=0.99,
        ent_coef=0.06,
    )

    raw_ppo_lstm_hyperparams = PPOLSTMHyperparams(
        policy="MlpLstmPolicy",
        learning_rate=3e-4,
        n_epochs=12,
        batch_size=32,
        n_steps=2048,
        gae_lambda=0.99,
        ent_coef=0.06,
    )


    print(f"{reward_mode} PPO agent initiated")
    ppo_agent = PPO_Agent(ppo_env, h_params=raw_ppo_hyperparams)
    ppo_lstm_agent = PPO_LSTM_Agent(ppo_env, raw_ppo_lstm_hyperparams)
    ppo_agent.learn(timesteps=TIMESTEPS, pbar=True)
    ppo_lstm_agent.learn(timesteps=TIMESTEPS, pbar=True)


    # ----- DQN env (discrete actions) -----
    dqn_env = DiscretePortfolioEnv(
        train_r.loc[train_f.index],
        train_f,
        window=WINDOW,
        cost_bps=COST_BPS,
        reward_mode=reward_mode,
        lambda_risk=lambda_risk,
        vol_window=vol_window,
    )

    dqn_hparams = DQNHyperparams()  # use your defaults
    print(f"{reward_mode} DQN agent initiated")
    dqn_agent = DQN_Agent(dqn_env, h_params=dqn_hparams)
    dqn_agent.learn(timesteps=TIMESTEPS, pbar=True)

    return ppo_agent, dqn_agent, ppo_lstm_agent


def eval_agent(agent, env):
    """
    Run a full episode for a given agent/env and return list of per-step rewards.
    """
    obs, _ = env.reset()
    rets, done = [], False
    while not done:
        action = agent.make_action(obs, deterministic=True)
        obs, r, done, _, _ = env.step(action)
        rets.append(r)
    return rets


# -------------------- DAILY / RAW -------------------- #

def run_train_test_daily_raw():
    # 1) load prices (train/test, daily)
    train_prices = load_split_freq(split="train", freq="daily")
    test_prices = load_split_freq(split="test", freq="daily")

    # 2) log returns + features
    train_r, test_r = to_log_returns(train_prices), to_log_returns(test_prices)
    train_f, test_f = build_features(train_r, WINDOW), build_features(test_r, WINDOW)

    # 3) train PPO + DQN (raw reward)
    ppo_agent, dqn_agent, ppo_lstm_agent = train_ppo_and_dqn(
        train_r,
        train_f,
        reward_mode="raw",
        lambda_risk=0.0,
    )

    # 4) test envs (raw reward)
    ppo_test_env = PortfolioEnv(
        test_r.loc[test_f.index],
        test_f,
        window=WINDOW,
        cost_bps=COST_BPS,
        reward_mode="raw",
        lambda_risk=0.0,
    )
    dqn_test_env = DiscretePortfolioEnv(
        test_r.loc[test_f.index],
        test_f,
        window=WINDOW,
        cost_bps=COST_BPS,
        reward_mode="raw",
        lambda_risk=0.0,
    )

    ppo_rets = eval_agent(ppo_agent, ppo_test_env)
    dqn_rets = eval_agent(dqn_agent, dqn_test_env)
    ppo_lstm_rets = eval_agent(ppo_lstm_agent, ppo_test_env)

    # align with dates: drop first WINDOW steps
    idx = test_f.index[WINDOW:]
    ppo_ret = pd.Series(ppo_rets, index=idx)
    dqn_ret = pd.Series(dqn_rets, index=idx)
    ppo_lstm_ret = pd.Series(ppo_lstm_rets, index=idx)

    ppo_cum = ppo_ret.cumsum()
    dqn_cum = dqn_ret.cumsum()
    ppo_lstm_cum = ppo_lstm_ret.cumsum()

    # baselines on the same period
    test_slice = test_r.loc[test_f.index].iloc[WINDOW:]
    ew_r, ew_cum = equal_weight(test_slice, freq="M", cost_bps=20)
    bh_r, bh_cum = buy_and_hold(test_slice)

    # metrics print
    print("Metrics (DAILY, RAW reward):")
    print("  PPO:", ann_metrics(ppo_ret))
    print("  PPO_LSTM", ann_metrics(ppo_lstm_ret))
    print("  DQN:", ann_metrics(dqn_ret))
    print("  EW :", ann_metrics(ew_r))
    print("  BH :", ann_metrics(bh_r))

    # equity curve figure
    plot_equity(
        {"PPO_raw": ppo_cum, "PPO_LSTM_raw": ppo_lstm_cum, "DQN_raw": dqn_cum, "EW": ew_cum, "BH": bh_cum},
        "results/figures/equity_daily_raw.png",
    )
    print("Saved figure -> results/figures/equity_daily_raw.png")

    # metrics CSV
    metrics_dict = {
        "PPO": ann_metrics(ppo_ret),
        "DQN": ann_metrics(dqn_ret),
        "PPO_LSTM": ann_metrics(ppo_lstm_ret), 
        "EW": ann_metrics(ew_r),
        "BH": ann_metrics(bh_r),
    }
    metrics_path = "results/metrics/raw_daily_metrics.csv"
    save_equity_to_csv(metrics_dict, metrics_path)

    # accumulated equity CSV
    equity_daily_dict = {
        "PPO": ppo_cum,
        "DQN": dqn_cum,
        "PPO_LSTM": ppo_lstm_cum, 
        "EW": ew_cum,
        "BH": bh_cum,
    }
    equity_daily_path = "results/metrics/raw_daily_accumalated_equity.csv"
    save_cum_ret_to_csv(equity_daily_dict, equity_daily_path)

    return None


# -------------------- DAILY / RISK-ADJUSTED -------------------- #

def run_train_test_daily_risk():
    # 1) load prices (train/test, daily)
    train_prices = load_split_freq(split="train", freq="daily")
    test_prices = load_split_freq(split="test", freq="daily")

    train_r, test_r = to_log_returns(train_prices), to_log_returns(test_prices)
    train_f, test_f = build_features(train_r, WINDOW), build_features(test_r, WINDOW)

    lambda_risk = 0.02
    vol_window = 10

    # 2) train PPO + DQN (risk-adjusted reward)
    ppo_agent, dqn_agent, ppo_lstm_agent = train_ppo_and_dqn(
        train_r,
        train_f,
        reward_mode="risk",
        lambda_risk=lambda_risk,
        vol_window=vol_window,
    )

    # 3) test envs (risk mode)
    ppo_test_env = PortfolioEnv(
        test_r.loc[test_f.index],
        test_f,
        window=WINDOW,
        cost_bps=COST_BPS,
        reward_mode="risk",
        lambda_risk=lambda_risk,
        vol_window=vol_window,
    )
    dqn_test_env = DiscretePortfolioEnv(
        test_r.loc[test_f.index],
        test_f,
        window=WINDOW,
        cost_bps=COST_BPS,
        reward_mode="risk",
        lambda_risk=lambda_risk,
        vol_window=vol_window,
    )

    ppo_rets = eval_agent(ppo_agent, ppo_test_env)
    dqn_rets = eval_agent(dqn_agent, dqn_test_env)
    ppo_lstm_rets = eval_agent(ppo_lstm_agent, ppo_test_env)

    idx = test_f.index[WINDOW:]
    ppo_ret = pd.Series(ppo_rets, index=idx)
    dqn_ret = pd.Series(dqn_rets, index=idx)
    ppo_lstm_ret = pd.Series(ppo_lstm_rets, index=idx)

    ppo_cum = ppo_ret.cumsum()
    dqn_cum = dqn_ret.cumsum()
    ppo_lstm_cum = ppo_lstm_ret.cumsum()

    # baselines
    test_slice = test_r.loc[test_f.index].iloc[WINDOW:]
    ew_r, ew_cum = equal_weight(test_slice, freq="M", cost_bps=20)
    bh_r, bh_cum = buy_and_hold(test_slice)

    print("Metrics (DAILY, RISK-ADJUSTED reward):")
    print("  PPO_risk:", ann_metrics(ppo_ret))
    print("  DQN_risk:", ann_metrics(dqn_ret))
    print("  PPO_LSTM_risk:", ann_metrics(ppo_lstm_ret))
    print("  EW      :", ann_metrics(ew_r))
    print("  BH      :", ann_metrics(bh_r))

    plot_equity(
        {"PPO_risk": ppo_cum, "DQN_risk": dqn_cum, "EW": ew_cum, "BH": bh_cum},
        "results/figures/equity_daily_risk.png",
    )
    print("Saved figure -> results/figures/equity_daily_risk.png")

    metrics_dict = {
        "PPO": ann_metrics(ppo_ret),
        "DQN": ann_metrics(dqn_ret),
        "EW": ann_metrics(ew_r),
        "BH": ann_metrics(bh_r),
    }
    metrics_path = "results/metrics/risk_daily_metrics.csv"
    save_equity_to_csv(metrics_dict, metrics_path)

    equity_daily_dict = {
        "PPO": ppo_cum,
        "DQN": dqn_cum,
        "EW": ew_cum,
        "BH": bh_cum,
    }
    equity_daily_path = "results/metrics/risk_daily_accumalated_equity.csv"
    save_cum_ret_to_csv(equity_daily_dict, equity_daily_path)

    return None


# -------------------- HOURLY / RAW -------------------- #

def run_train_test_hourly_raw():
    train_prices = load_split_freq(split="train", freq="hourly")
    test_prices = load_split_freq(split="test", freq="hourly")

    train_r, test_r = to_log_returns(train_prices), to_log_returns(test_prices)
    train_f, test_f = build_features(train_r, WINDOW), build_features(test_r, WINDOW)

    # train PPO + DQN (raw reward)
    ppo_agent, dqn_agent = train_ppo_and_dqn(
        train_r,
        train_f,
        reward_mode="raw",
        lambda_risk=0.0,
    )

    # test envs
    ppo_test_env = PortfolioEnv(
        test_r.loc[test_f.index],
        test_f,
        window=WINDOW,
        cost_bps=COST_BPS,
        reward_mode="raw",
        lambda_risk=0.0,
    )
    dqn_test_env = DiscretePortfolioEnv(
        test_r.loc[test_f.index],
        test_f,
        window=WINDOW,
        cost_bps=COST_BPS,
        reward_mode="raw",
        lambda_risk=0.0,
    )

    ppo_rets = eval_agent(ppo_agent, ppo_test_env)
    dqn_rets = eval_agent(dqn_agent, dqn_test_env)

    idx = test_f.index[WINDOW:]
    ppo_ret = pd.Series(ppo_rets, index=idx)
    dqn_ret = pd.Series(dqn_rets, index=idx)
    ppo_cum = ppo_ret.cumsum()
    dqn_cum = dqn_ret.cumsum()

    test_slice = test_r.loc[test_f.index].iloc[WINDOW:]
    ew_r, ew_cum = equal_weight(test_slice, freq="M", cost_bps=20)
    bh_r, bh_cum = buy_and_hold(test_slice)

    print("Metrics (HOURLY, RAW reward):")
    print("  PPO:", ann_metrics(ppo_ret))
    print("  DQN:", ann_metrics(dqn_ret))
    print("  EW :", ann_metrics(ew_r))
    print("  BH :", ann_metrics(bh_r))

    plot_equity(
        {"PPO_raw": ppo_cum, "DQN_raw": dqn_cum, "EW": ew_cum, "BH": bh_cum},
        "results/figures/equity_hourly_raw.png",
    )
    print("Saved figure -> results/figures/equity_hourly_raw.png")

    metrics_dict = {
        "PPO": ann_metrics(ppo_ret),
        "DQN": ann_metrics(dqn_ret),
        "EW": ann_metrics(ew_r),
        "BH": ann_metrics(bh_r),
    }
    metrics_path = "results/metrics/raw_hourly_metrics.csv"
    save_equity_to_csv(metrics_dict, metrics_path)

    equity_dict = {
        "PPO": ppo_cum,
        "DQN": dqn_cum,
        "EW": ew_cum,
        "BH": bh_cum,
    }
    equity_path = "results/metrics/raw_hourly_accumalated_equity.csv"
    save_cum_ret_to_csv(equity_dict, equity_path)

    return None


# -------------------- HOURLY / RISK-ADJUSTED -------------------- #

def run_train_test_hourly_risk():
    train_prices = load_split_freq(split="train", freq="hourly")
    test_prices = load_split_freq(split="test", freq="hourly")

    train_r, test_r = to_log_returns(train_prices), to_log_returns(test_prices)
    train_f, test_f = build_features(train_r, WINDOW), build_features(test_r, WINDOW)

    lambda_risk = 0.02
    vol_window = 15

    # train PPO + DQN (risk-adjusted)
    ppo_agent, dqn_agent = train_ppo_and_dqn(
        train_r,
        train_f,
        reward_mode="risk",
        lambda_risk=lambda_risk,
        vol_window=vol_window,
    )

    ppo_test_env = PortfolioEnv(
        test_r.loc[test_f.index],
        test_f,
        window=WINDOW,
        cost_bps=COST_BPS,
        reward_mode="risk",
        lambda_risk=lambda_risk,
        vol_window=vol_window,
    )
    dqn_test_env = DiscretePortfolioEnv(
        test_r.loc[test_f.index],
        test_f,
        window=WINDOW,
        cost_bps=COST_BPS,
        reward_mode="risk",
        lambda_risk=lambda_risk,
        vol_window=vol_window,
    )

    ppo_rets = eval_agent(ppo_agent, ppo_test_env)
    dqn_rets = eval_agent(dqn_agent, dqn_test_env)

    idx = test_f.index[WINDOW:]
    ppo_ret = pd.Series(ppo_rets, index=idx)
    dqn_ret = pd.Series(dqn_rets, index=idx)
    ppo_cum = ppo_ret.cumsum()
    dqn_cum = dqn_ret.cumsum()

    test_slice = test_r.loc[test_f.index].iloc[WINDOW:]
    ew_r, ew_cum = equal_weight(test_slice, freq="M", cost_bps=20)
    bh_r, bh_cum = buy_and_hold(test_slice)

    print("Metrics (HOURLY, RISK-ADJUSTED reward):")
    print("  PPO_risk:", ann_metrics(ppo_ret))
    print("  DQN_risk:", ann_metrics(dqn_ret))
    print("  EW      :", ann_metrics(ew_r))
    print("  BH      :", ann_metrics(bh_r))

    plot_equity(
        {"PPO_risk": ppo_cum, "DQN_risk": dqn_cum, "EW": ew_cum, "BH": bh_cum},
        "results/figures/equity_hourly_risk.png",
    )
    print("Saved figure -> results/figures/equity_hourly_risk.png")

    metrics_dict = {
        "PPO": ann_metrics(ppo_ret),
        "DQN": ann_metrics(dqn_ret),
        "EW": ann_metrics(ew_r),
        "BH": ann_metrics(bh_r),
    }
    metrics_path = "results/metrics/risk_hourly_metrics.csv"
    save_equity_to_csv(metrics_dict, metrics_path)

    equity_dict = {
        "PPO": ppo_cum,
        "DQN": dqn_cum,
        "EW": ew_cum,
        "BH": bh_cum,
    }
    equity_path = "results/metrics/risk_hourly_accumalated_equity.csv"
    save_cum_ret_to_csv(equity_dict, equity_path)

    return None

# -------------------- 30min / RAW -------------------- #

def run_train_test_30min_raw():
    train_prices = load_split_freq(split="train", freq="30min")
    test_prices = load_split_freq(split="test", freq="30min")

    train_r, test_r = to_log_returns(train_prices), to_log_returns(test_prices)
    train_f, test_f = build_features(train_r, WINDOW), build_features(test_r, WINDOW)

    # train PPO + DQN (raw reward)
    ppo_agent, dqn_agent = train_ppo_and_dqn(
        train_r,
        train_f,
        reward_mode="raw",
        lambda_risk=0.0,
    )

    # test envs
    ppo_test_env = PortfolioEnv(
        test_r.loc[test_f.index],
        test_f,
        window=WINDOW,
        cost_bps=COST_BPS,
        reward_mode="raw",
        lambda_risk=0.0,
    )
    dqn_test_env = DiscretePortfolioEnv(
        test_r.loc[test_f.index],
        test_f,
        window=WINDOW,
        cost_bps=COST_BPS,
        reward_mode="raw",
        lambda_risk=0.0,
    )

    ppo_rets = eval_agent(ppo_agent, ppo_test_env)
    dqn_rets = eval_agent(dqn_agent, dqn_test_env)

    idx = test_f.index[WINDOW:]
    ppo_ret = pd.Series(ppo_rets, index=idx)
    dqn_ret = pd.Series(dqn_rets, index=idx)
    ppo_cum = ppo_ret.cumsum()
    dqn_cum = dqn_ret.cumsum()

    test_slice = test_r.loc[test_f.index].iloc[WINDOW:]
    ew_r, ew_cum = equal_weight(test_slice, freq="M", cost_bps=20)
    bh_r, bh_cum = buy_and_hold(test_slice)

    print("Metrics (30min, RAW reward):")
    print("  PPO:", ann_metrics(ppo_ret))
    print("  DQN:", ann_metrics(dqn_ret))
    print("  EW :", ann_metrics(ew_r))
    print("  BH :", ann_metrics(bh_r))

    plot_equity(
        {"PPO_raw": ppo_cum, "DQN_raw": dqn_cum, "EW": ew_cum, "BH": bh_cum},
        "results/figures/equity_30min_raw.png",
    )
    print("Saved figure -> results/figures/equity_30min_raw.png")

    metrics_dict = {
        "PPO": ann_metrics(ppo_ret),
        "DQN": ann_metrics(dqn_ret),
        "EW": ann_metrics(ew_r),
        "BH": ann_metrics(bh_r),
    }
    metrics_path = "results/metrics/raw_30min_metrics.csv"
    save_equity_to_csv(metrics_dict, metrics_path)

    equity_dict = {
        "PPO": ppo_cum,
        "DQN": dqn_cum,
        "EW": ew_cum,
        "BH": bh_cum,
    }
    equity_path = "results/metrics/raw_30min_accumalated_equity.csv"
    save_cum_ret_to_csv(equity_dict, equity_path)

    return None


# -------------------- 30min / RISK-ADJUSTED -------------------- #

def run_train_test_30min_risk():
    train_prices = load_split_freq(split="train", freq="30min")
    test_prices = load_split_freq(split="test", freq="30min")

    train_r, test_r = to_log_returns(train_prices), to_log_returns(test_prices)
    train_f, test_f = build_features(train_r, WINDOW), build_features(test_r, WINDOW)

    lambda_risk = 0.02
    vol_window = 60

    # train PPO + DQN (risk-adjusted)
    ppo_agent, dqn_agent = train_ppo_and_dqn(
        train_r,
        train_f,
        reward_mode="risk",
        lambda_risk=lambda_risk,
        vol_window=vol_window,
    )

    ppo_test_env = PortfolioEnv(
        test_r.loc[test_f.index],
        test_f,
        window=WINDOW,
        cost_bps=COST_BPS,
        reward_mode="risk",
        lambda_risk=lambda_risk,
        vol_window=vol_window,
    )
    dqn_test_env = DiscretePortfolioEnv(
        test_r.loc[test_f.index],
        test_f,
        window=WINDOW,
        cost_bps=COST_BPS,
        reward_mode="risk",
        lambda_risk=lambda_risk,
        vol_window=vol_window,
    )

    ppo_rets = eval_agent(ppo_agent, ppo_test_env)
    dqn_rets = eval_agent(dqn_agent, dqn_test_env)

    idx = test_f.index[WINDOW:]
    ppo_ret = pd.Series(ppo_rets, index=idx)
    dqn_ret = pd.Series(dqn_rets, index=idx)
    ppo_cum = ppo_ret.cumsum()
    dqn_cum = dqn_ret.cumsum()

    test_slice = test_r.loc[test_f.index].iloc[WINDOW:]
    ew_r, ew_cum = equal_weight(test_slice, freq="M", cost_bps=20)
    bh_r, bh_cum = buy_and_hold(test_slice)

    print("Metrics (30min, RISK-ADJUSTED reward):")
    print("  PPO_risk:", ann_metrics(ppo_ret))
    print("  DQN_risk:", ann_metrics(dqn_ret))
    print("  EW      :", ann_metrics(ew_r))
    print("  BH      :", ann_metrics(bh_r))

    plot_equity(
        {"PPO_risk": ppo_cum, "DQN_risk": dqn_cum, "EW": ew_cum, "BH": bh_cum},
        "results/figures/equity_30min_risk.png",
    )
    print("Saved figure -> results/figures/equity_30min_risk.png")

    metrics_dict = {
        "PPO": ann_metrics(ppo_ret),
        "DQN": ann_metrics(dqn_ret),
        "EW": ann_metrics(ew_r),
        "BH": ann_metrics(bh_r),
    }
    metrics_path = "results/metrics/risk_30min_metrics.csv"
    save_equity_to_csv(metrics_dict, metrics_path)

    equity_dict = {
        "PPO": ppo_cum,
        "DQN": dqn_cum,
        "EW": ew_cum,
        "BH": bh_cum,
    }
    equity_path = "results/metrics/risk_30min_accumalated_equity.csv"
    save_cum_ret_to_csv(equity_dict, equity_path)

    return None

# -------------------- main -------------------- #

if __name__ == "__main__":
    print(torch.backends.mps.is_available(), torch.backends.mps.is_built())
    run_train_test_daily_raw()
    # run_train_test_daily_risk()
    # run_train_test_hourly_raw()
    # run_train_test_hourly_risk()
    # run_train_test_30min_raw()
    # run_train_test_30min_risk()
