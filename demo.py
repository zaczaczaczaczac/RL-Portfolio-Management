# demo.py
from stable_baselines3 import PPO
import pandas as pd

from src.data_loader import load_split_freq
from src.features import to_log_returns, build_features
from src.envs import PortfolioEnv
from src.baselines import buy_and_hold, equal_weight
from src.agents.ppo_agent import PPO_Agent, PPOHyperparams
from src.evaluate import ann_metrics, plot_equity, save_equity_to_csv, save_cum_ret_to_csv

WINDOW = 20
COST_BPS = 20
TIMESTEPS = 350_000  

# daily
def run_train_test_daily_raw():
    train_prices = load_split_freq(split="train", freq="daily")
    test_prices  = load_split_freq(split="test",  freq="daily")

    train_r, test_r = to_log_returns(train_prices), to_log_returns(test_prices)
    train_f, test_f = build_features(train_r, WINDOW), build_features(test_r, WINDOW)

    env = PortfolioEnv(
        train_r.loc[train_f.index],
        train_f,
        window=WINDOW,
        cost_bps=COST_BPS,
        reward_mode="raw",      # raw mode
        lambda_risk=0.0,
    )
    # model = PPO("MlpPolicy", env, verbose=0)
    # model.learn(total_timesteps=TIMESTEPS)

    # init PPO agent and hyperparams
    raw_ppo_hyperparams = PPOHyperparams(
        policy="MlpPolicy",
        learning_rate=4e-4, 
        n_epochs=10, 
        batch_size=32,
        n_steps=2048,
        gae_lambda=0.99,
        ent_coef=0.05
    )

    print("raw ppo agent initiated")
    raw_PPO_agent = PPO_Agent(env, h_params=raw_ppo_hyperparams)
    raw_PPO_agent.learn(timesteps=TIMESTEPS, pbar=True)

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
        action = raw_PPO_agent.make_action(obs, deterministic=True)
        obs, r, done, _, _ = test_env.step(action)
        rets.append(r)

    idx = test_f.index[WINDOW:]          
    rl_ret = pd.Series(rets, index=idx)
    rl_cum = rl_ret.cumsum()

    test_slice = test_r.loc[test_f.index].iloc[WINDOW:]
    ew_r, ew_cum = equal_weight(test_slice, freq='M', cost_bps=20)
    bh_r, bh_cum = buy_and_hold(test_slice)


    # record results as plot and csv
    print("Metrics (RAW reward):")
    print("  PPO:", ann_metrics(rl_ret))
    print("  EW :", ann_metrics(ew_r))
    print("  BH :", ann_metrics(bh_r))

    plot_equity(
        {"RL_PPO_raw": rl_cum, "EW": ew_cum, "BH": bh_cum},
        "results/figures/equity_daily_raw.png"
    )
    print("Saved figure -> results/figures/equity_daily_raw.png")

    metrics_dict = dict()
    metrics_dict["PPO"] = ann_metrics(rl_ret)
    metrics_dict["DQN"] = ann_metrics(rl_ret) # temp placeholder
    metrics_dict["EW"] = ann_metrics(ew_r)
    metrics_dict["BH"] = ann_metrics(bh_r)
    metrics_path = "results/metrics/raw_daily_metrics.csv"
    save_equity_to_csv(metrics_dict, metrics_path)

    equity_daily_dict = dict()
    equity_daily_dict["PPO"] = rl_cum
    equity_daily_dict["DQN"] = rl_cum  # temp placeholder
    equity_daily_dict["EW"] = ew_cum
    equity_daily_dict["BH"] = bh_cum
    equity_daily_path = "results/metrics/raw_daily_accumalated_equity.csv"
    save_cum_ret_to_csv(equity_daily_dict, equity_daily_path)
    return None

def run_train_test_daily_risk():
    train_prices = load_split_freq(split="train", freq="daily")
    test_prices  = load_split_freq(split="test",  freq="daily")

    train_r, test_r = to_log_returns(train_prices), to_log_returns(test_prices)
    train_f, test_f = build_features(train_r, WINDOW), build_features(test_r, WINDOW)

    env = PortfolioEnv(
        train_r.loc[train_f.index],
        train_f,
        window=WINDOW,
        cost_bps=COST_BPS,
        reward_mode="risk",    
        lambda_risk=0.02,       
        vol_window=15,
    )
    # model = PPO("MlpPolicy", env, verbose=0)
    # model.learn(total_timesteps=TIMESTEPS)
    risk_ppo_hyperparams = PPOHyperparams(
        policy="MlpPolicy",
        learning_rate=4e-4, 
        n_epochs=10, 
        batch_size=32,
        n_steps=2048,
        gae_lambda=0.99,
        ent_coef=0.05
    )

    print("risk ppo agent initiated")
    risk_PPO_agent = PPO_Agent(env, h_params=risk_ppo_hyperparams)
    risk_PPO_agent.learn(timesteps=TIMESTEPS, pbar=True)

    test_env = PortfolioEnv(
        test_r.loc[test_f.index],
        test_f,
        window=WINDOW,
        cost_bps=COST_BPS,
        reward_mode="risk",
        lambda_risk=0.02,
        vol_window=15,
    )
    obs, _ = test_env.reset()
    rets, done = [], False
    while not done:
        action = risk_PPO_agent.make_action(obs, deterministic=True)
        obs, r, done, _, _ = test_env.step(action)
        rets.append(r)

    idx = test_f.index[WINDOW:]
    rl_ret = pd.Series(rets, index=idx)
    rl_cum = rl_ret.cumsum()

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

    metrics_dict = dict()
    metrics_dict["PPO"] = ann_metrics(rl_ret)
    metrics_dict["DQN"] = ann_metrics(rl_ret) # temp placeholder
    metrics_dict["EW"] = ann_metrics(ew_r)
    metrics_dict["BH"] = ann_metrics(bh_r)
    metrics_path = "results/metrics/risk_daily_metrics.csv"
    save_equity_to_csv(metrics_dict, metrics_path)

    equity_daily_dict = dict()
    equity_daily_dict["PPO"] = rl_cum
    equity_daily_dict["DQN"] = rl_cum  # temp placeholder
    equity_daily_dict["EW"] = ew_cum
    equity_daily_dict["BH"] = bh_cum
    equity_daily_path = "results/metrics/risk_daily_accumalated_equity.csv"
    save_cum_ret_to_csv(equity_daily_dict, equity_daily_path)
    return None

# hourly
def run_train_test_hourly_raw():
    train_prices = load_split_freq(split="train", freq="hourly")
    test_prices  = load_split_freq(split="test",  freq="hourly")

    train_r, test_r = to_log_returns(train_prices), to_log_returns(test_prices)
    train_f, test_f = build_features(train_r, WINDOW), build_features(test_r, WINDOW)

    env = PortfolioEnv(
        train_r.loc[train_f.index],
        train_f,
        window=WINDOW,
        cost_bps=COST_BPS,
        reward_mode="raw",      # raw mode
        lambda_risk=0.0,
    )
    # model = PPO("MlpPolicy", env, verbose=0)
    # model.learn(total_timesteps=TIMESTEPS)

    # init PPO agent and hyperparams
    raw_ppo_hyperparams = PPOHyperparams(
        policy="MlpPolicy",
        learning_rate=4e-4, 
        n_epochs=10, 
        batch_size=32,
        n_steps=2048,
        gae_lambda=0.99,
        ent_coef=0.05
    )

    print("raw ppo agent initiated")
    raw_PPO_agent = PPO_Agent(env, h_params=raw_ppo_hyperparams)
    raw_PPO_agent.learn(timesteps=TIMESTEPS, pbar=True)

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
        action = raw_PPO_agent.make_action(obs, deterministic=True)
        obs, r, done, _, _ = test_env.step(action)
        rets.append(r)

    idx = test_f.index[WINDOW:]       
    rl_ret = pd.Series(rets, index=idx)
    rl_cum = rl_ret.cumsum()

    test_slice = test_r.loc[test_f.index].iloc[WINDOW:]
    ew_r, ew_cum = equal_weight(test_slice, freq='M', cost_bps=20)
    bh_r, bh_cum = buy_and_hold(test_slice)


    # record results as plot and csv
    print("Metrics (RAW reward):")
    print("  PPO:", ann_metrics(rl_ret))
    print("  EW :", ann_metrics(ew_r))
    print("  BH :", ann_metrics(bh_r))

    plot_equity(
        {"RL_PPO_raw": rl_cum, "EW": ew_cum, "BH": bh_cum},
        "results/figures/equity_hourly_raw.png"
    )
    print("Saved figure -> results/figures/equity_hourly_raw.png")

    metrics_dict = dict()
    metrics_dict["PPO"] = ann_metrics(rl_ret)
    metrics_dict["DQN"] = ann_metrics(rl_ret) # temp placeholder
    metrics_dict["EW"] = ann_metrics(ew_r)
    metrics_dict["BH"] = ann_metrics(bh_r)
    metrics_path = "results/metrics/raw_hourly_metrics.csv"
    save_equity_to_csv(metrics_dict, metrics_path)

    equity_daily_dict = dict()
    equity_daily_dict["PPO"] = rl_cum
    equity_daily_dict["DQN"] = rl_cum  # temp placeholder
    equity_daily_dict["EW"] = ew_cum
    equity_daily_dict["BH"] = bh_cum
    equity_daily_path = "results/metrics/raw_hourly_accumalated_equity.csv"
    save_cum_ret_to_csv(equity_daily_dict, equity_daily_path)
    return None

def run_train_test_hourly_risk():
    train_prices = load_split_freq(split="train", freq="hourly")
    test_prices  = load_split_freq(split="test",  freq="hourly")

    train_r, test_r = to_log_returns(train_prices), to_log_returns(test_prices)
    train_f, test_f = build_features(train_r, WINDOW), build_features(test_r, WINDOW)

    env = PortfolioEnv(
        train_r.loc[train_f.index],
        train_f,
        window=WINDOW,
        cost_bps=COST_BPS,
        reward_mode="risk",    
        lambda_risk=0.02,       
        vol_window=15,
    )
    # model = PPO("MlpPolicy", env, verbose=0)
    # model.learn(total_timesteps=TIMESTEPS)
    risk_ppo_hyperparams = PPOHyperparams(
        policy="MlpPolicy",
        learning_rate=4e-4, 
        n_epochs=10, 
        batch_size=32,
        n_steps=2048,
        gae_lambda=0.99,
        ent_coef=0.05
    )

    print("risk ppo agent initiated")
    risk_PPO_agent = PPO_Agent(env, h_params=risk_ppo_hyperparams)
    risk_PPO_agent.learn(timesteps=TIMESTEPS, pbar=True)

    test_env = PortfolioEnv(
        test_r.loc[test_f.index],
        test_f,
        window=WINDOW,
        cost_bps=COST_BPS,
        reward_mode="risk",
        lambda_risk=0.02,
        vol_window=15,
    )
    obs, _ = test_env.reset()
    rets, done = [], False
    while not done:
        action = risk_PPO_agent.make_action(obs, deterministic=True)
        obs, r, done, _, _ = test_env.step(action)
        rets.append(r)

    idx = test_f.index[WINDOW:]
    rl_ret = pd.Series(rets, index=idx)
    rl_cum = rl_ret.cumsum()

    test_slice = test_r.loc[test_f.index].iloc[WINDOW:]
    ew_r, ew_cum = equal_weight(test_slice, freq='M', cost_bps=20)
    bh_r, bh_cum = buy_and_hold(test_slice)

    print("Metrics (RISK-ADJUSTED reward):")
    print("  PPO_risk:", ann_metrics(rl_ret))
    print("  EW      :", ann_metrics(ew_r))
    print("  BH      :", ann_metrics(bh_r))

    plot_equity(
        {"RL_PPO_risk": rl_cum, "EW": ew_cum, "BH": bh_cum},
        "results/figures/equity_hourly_risk.png"
    )
    print("Saved figure -> results/figures/equity_hourly_risk.png")

    metrics_dict = dict()
    metrics_dict["PPO"] = ann_metrics(rl_ret)
    metrics_dict["DQN"] = ann_metrics(rl_ret) # temp placeholder
    metrics_dict["EW"] = ann_metrics(ew_r)
    metrics_dict["BH"] = ann_metrics(bh_r)
    metrics_path = "results/metrics/risk_hourly_metrics.csv"
    save_equity_to_csv(metrics_dict, metrics_path)

    equity_daily_dict = dict()
    equity_daily_dict["PPO"] = rl_cum
    equity_daily_dict["DQN"] = rl_cum  # temp placeholder
    equity_daily_dict["EW"] = ew_cum
    equity_daily_dict["BH"] = bh_cum
    equity_daily_path = "results/metrics/risk_hourly_accumalated_equity.csv"
    save_cum_ret_to_csv(equity_daily_dict, equity_daily_path)
    return None

# 30 mins
def run_train_test_30min_raw():
    train_prices = load_split_freq(split="train", freq="30min")
    test_prices  = load_split_freq(split="test",  freq="30min")

    train_r, test_r = to_log_returns(train_prices), to_log_returns(test_prices)
    train_f, test_f = build_features(train_r, WINDOW), build_features(test_r, WINDOW)

    env = PortfolioEnv(
        train_r.loc[train_f.index],
        train_f,
        window=WINDOW,
        cost_bps=COST_BPS,
        reward_mode="raw",      # raw mode
        lambda_risk=0.0,
    )
    # model = PPO("MlpPolicy", env, verbose=0)
    # model.learn(total_timesteps=TIMESTEPS)

    # init PPO agent and hyperparams
    raw_ppo_hyperparams = PPOHyperparams(
        policy="MlpPolicy",
        learning_rate=4e-4, 
        n_epochs=10, 
        batch_size=32,
        n_steps=2048,
        gae_lambda=0.99,
        ent_coef=0.05
    )

    print("raw ppo agent initiated")
    raw_PPO_agent = PPO_Agent(env, h_params=raw_ppo_hyperparams)
    raw_PPO_agent.learn(timesteps=TIMESTEPS, pbar=True)

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
        action = raw_PPO_agent.make_action(obs, deterministic=True)
        obs, r, done, _, _ = test_env.step(action)
        rets.append(r)

    idx = test_f.index[WINDOW:]         
    rl_ret = pd.Series(rets, index=idx)
    rl_cum = rl_ret.cumsum()

    test_slice = test_r.loc[test_f.index].iloc[WINDOW:]
    ew_r, ew_cum = equal_weight(test_slice, freq='M', cost_bps=20)
    bh_r, bh_cum = buy_and_hold(test_slice)


    # record results as plot and csv
    print("Metrics (RAW reward):")
    print("  PPO:", ann_metrics(rl_ret))
    print("  EW :", ann_metrics(ew_r))
    print("  BH :", ann_metrics(bh_r))

    plot_equity(
        {"RL_PPO_raw": rl_cum, "EW": ew_cum, "BH": bh_cum},
        "results/figures/equity_30min_raw.png"
    )
    print("Saved figure -> results/figures/equity_30min_raw.png")

    metrics_dict = dict()
    metrics_dict["PPO"] = ann_metrics(rl_ret)
    metrics_dict["DQN"] = ann_metrics(rl_ret) # temp placeholder
    metrics_dict["EW"] = ann_metrics(ew_r)
    metrics_dict["BH"] = ann_metrics(bh_r)
    metrics_path = "results/metrics/raw_30min_metrics.csv"
    save_equity_to_csv(metrics_dict, metrics_path)

    equity_daily_dict = dict()
    equity_daily_dict["PPO"] = rl_cum
    equity_daily_dict["DQN"] = rl_cum  # temp placeholder
    equity_daily_dict["EW"] = ew_cum
    equity_daily_dict["BH"] = bh_cum
    equity_daily_path = "results/metrics/raw_30min_accumalated_equity.csv"
    save_cum_ret_to_csv(equity_daily_dict, equity_daily_path)
    return None

def run_train_test_30min_risk():
    train_prices = load_split_freq(split="train", freq="30min")
    test_prices  = load_split_freq(split="test",  freq="30min")

    train_r, test_r = to_log_returns(train_prices), to_log_returns(test_prices)
    train_f, test_f = build_features(train_r, WINDOW), build_features(test_r, WINDOW)

    env = PortfolioEnv(
        train_r.loc[train_f.index],
        train_f,
        window=WINDOW,
        cost_bps=COST_BPS,
        reward_mode="risk",    
        lambda_risk=0.02,      
        vol_window=15,
    )
    # model = PPO("MlpPolicy", env, verbose=0)
    # model.learn(total_timesteps=TIMESTEPS)
    risk_ppo_hyperparams = PPOHyperparams(
        policy="MlpPolicy",
        learning_rate=4e-4, 
        n_epochs=10, 
        batch_size=32,
        n_steps=2048,
        gae_lambda=0.99,
        ent_coef=0.05
    )

    print("risk ppo agent initiated")
    risk_PPO_agent = PPO_Agent(env, h_params=risk_ppo_hyperparams)
    risk_PPO_agent.learn(timesteps=TIMESTEPS, pbar=True)

    test_env = PortfolioEnv(
        test_r.loc[test_f.index],
        test_f,
        window=WINDOW,
        cost_bps=COST_BPS,
        reward_mode="risk",
        lambda_risk=0.02,
        vol_window=15,
    )
    obs, _ = test_env.reset()
    rets, done = [], False
    while not done:
        action = risk_PPO_agent.make_action(obs, deterministic=True)
        obs, r, done, _, _ = test_env.step(action)
        rets.append(r)

    idx = test_f.index[WINDOW:]
    rl_ret = pd.Series(rets, index=idx)
    rl_cum = rl_ret.cumsum()

    test_slice = test_r.loc[test_f.index].iloc[WINDOW:]
    ew_r, ew_cum = equal_weight(test_slice, freq='M', cost_bps=20)
    bh_r, bh_cum = buy_and_hold(test_slice)

    print("Metrics (RISK-ADJUSTED reward):")
    print("  PPO_risk:", ann_metrics(rl_ret))
    print("  EW      :", ann_metrics(ew_r))
    print("  BH      :", ann_metrics(bh_r))

    plot_equity(
        {"RL_PPO_risk": rl_cum, "EW": ew_cum, "BH": bh_cum},
        "results/figures/equity_30min_risk.png"
    )
    print("Saved figure -> results/figures/equity_30min_risk.png")

    metrics_dict = dict()
    metrics_dict["PPO"] = ann_metrics(rl_ret)
    metrics_dict["DQN"] = ann_metrics(rl_ret) # temp placeholder
    metrics_dict["EW"] = ann_metrics(ew_r)
    metrics_dict["BH"] = ann_metrics(bh_r)
    metrics_path = "results/metrics/risk_30min_metrics.csv"
    save_equity_to_csv(metrics_dict, metrics_path)

    equity_daily_dict = dict()
    equity_daily_dict["PPO"] = rl_cum
    equity_daily_dict["DQN"] = rl_cum  # temp placeholder
    equity_daily_dict["EW"] = ew_cum
    equity_daily_dict["BH"] = bh_cum
    equity_daily_path = "results/metrics/risk_30min_accumalated_equity.csv"
    save_cum_ret_to_csv(equity_daily_dict, equity_daily_path)
    return None


if __name__ == "__main__":
    run_train_test_daily_raw()
    run_train_test_daily_risk()
    run_train_test_hourly_raw()
    run_train_test_hourly_risk()
    run_train_test_30min_raw()
    run_train_test_30min_risk()
    
