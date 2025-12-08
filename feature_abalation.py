# ablation_experiments.py
"""
Comprehensive ablation study for RL-Portfolio-Management.

Blocks:
  1) Feature ablation: F0–F3, PPO, RISK reward, freq ∈ {daily, 30min, hourly}
  2) Reward ablation: RAW vs RISK, PPO, best feature cfg, all freqs
  3) Agent ablation: PPO vs DQN, RISK reward, best feature cfg, all freqs

Run:
    conda activate tradingrl
    python ablation_experiments.py
"""

import os
import pandas as pd

from src.data_loader import load_split_freq
from src.features import to_log_returns, build_features
from src.envs import PortfolioEnv, DiscretePortfolioEnv
from src.baselines import buy_and_hold, equal_weight
from src.agents.ppo_agent import PPO_Agent, PPOHyperparams
from src.agents.dqn_agent import DQN_Agent, DQNHyperparams
from src.evaluate import ann_metrics, plot_equity, ann_freq_for

# -------------------- config -------------------- #

WINDOW = 20
COST_BPS = 20

# shorter for debugging; increase for final runs
TIMESTEPS_FEATURE = 200_000
TIMESTEPS_REWARD = 300_000
TIMESTEPS_AGENT  = 300_000

FEATURE_CONFIGS = ["F0", "F1", "F2", "F3"]
FREQUENCIES = ["daily", "hourly"] # comment out "30min" if too slow

# set after you inspect Block 1 results (start with "F2")
BEST_FEATURE_CFG = "F2"

OUT_DIR = "results/ablation"
os.makedirs(OUT_DIR, exist_ok=True)


# -------------------- helpers -------------------- #

def train_ppo(train_r, train_f, reward_mode, timesteps, lambda_risk=0.0, vol_window=20):
    env = PortfolioEnv(
        train_r.loc[train_f.index],
        train_f,
        window=WINDOW,
        cost_bps=COST_BPS,
        reward_mode=reward_mode,
        lambda_risk=lambda_risk,
        vol_window=vol_window,
    )

    hparams = PPOHyperparams(
        policy="MlpPolicy",
        learning_rate=4e-4,
        n_epochs=10,
        batch_size=32,
        n_steps=2048,
        gae_lambda=0.99,
        ent_coef=0.05,
    )

    agent = PPO_Agent(env, h_params=hparams)
    agent.learn(timesteps=timesteps, pbar=True)
    return agent


def train_dqn(train_r, train_f, reward_mode, timesteps, lambda_risk=0.0, vol_window=20):
    env = DiscretePortfolioEnv(
        train_r.loc[train_f.index],
        train_f,
        window=WINDOW,
        cost_bps=COST_BPS,
        reward_mode=reward_mode,
        lambda_risk=lambda_risk,
        vol_window=vol_window,
    )

    hparams = DQNHyperparams()
    agent = DQN_Agent(env, h_params=hparams)
    agent.learn(timesteps=timesteps, pbar=True)
    return agent


def rollout(agent, env):
    obs, _ = env.reset()
    rets, done = [], False
    while not done:
        action = agent.make_action(obs, deterministic=True)
        obs, r, done, _, _ = env.step(action)
        rets.append(r)
    return rets


def make_test_env(agent_type, test_r, test_f, reward_mode, lambda_risk=0.0, vol_window=20):
    if agent_type == "ppo":
        return PortfolioEnv(
            test_r.loc[test_f.index],
            test_f,
            window=WINDOW,
            cost_bps=COST_BPS,
            reward_mode=reward_mode,
            lambda_risk=lambda_risk,
            vol_window=vol_window,
        )
    else:
        return DiscretePortfolioEnv(
            test_r.loc[test_f.index],
            test_f,
            window=WINDOW,
            cost_bps=COST_BPS,
            reward_mode=reward_mode,
            lambda_risk=lambda_risk,
            vol_window=vol_window,
        )


def baselines_on_slice(test_r_slice: pd.DataFrame):
    ew_r, ew_cum = equal_weight(test_r_slice, freq="M", cost_bps=20)
    bh_r, bh_cum = buy_and_hold(test_r_slice)
    return ew_r, ew_cum, bh_r, bh_cum


# -------------------- Block 1: feature ablation (RISK reward) -------------------- #

def run_feature_ablation():
    """
    For each freq and feature cfg, train PPO with risk-adjusted reward.
    """
    rows = []

    for freq in FREQUENCIES:
        print(f"\n===== [Block 1] Feature ablation, freq={freq}, reward=RISK =====")

        train_prices = load_split_freq(split="train", freq=freq)
        test_prices = load_split_freq(split="test", freq=freq)
        train_r, test_r = to_log_returns(train_prices), to_log_returns(test_prices)

        ann_f = ann_freq_for(freq)

        for cfg in FEATURE_CONFIGS:
            print(f"\n--- cfg={cfg} ---")

            train_f = build_features(train_r, WINDOW, config=cfg)
            test_f  = build_features(test_r,  WINDOW, config=cfg)

            train_r_aligned = train_r.loc[train_f.index]
            test_r_aligned  = test_r.loc[test_f.index]

            lambda_risk = 0.02
            vol_window  = 15

            agent = train_ppo(
                train_r_aligned,
                train_f,
                reward_mode="risk",
                timesteps=TIMESTEPS_FEATURE,
                lambda_risk=lambda_risk,
                vol_window=vol_window,
            )

            test_env = make_test_env(
                "ppo",
                test_r_aligned,
                test_f,
                reward_mode="risk",
                lambda_risk=lambda_risk,
                vol_window=vol_window,
            )

            rets = rollout(agent, test_env)
            idx = test_f.index[WINDOW:]
            ret_s = pd.Series(rets, index=idx)
            cum_s = ret_s.cumsum()

            test_slice = test_r_aligned.iloc[WINDOW:]
            ew_r, ew_cum, bh_r, bh_cum = baselines_on_slice(test_slice)

            m_ppo = ann_metrics(ret_s, freq=ann_f)
            m_ew  = ann_metrics(ew_r,  freq=ann_f)
            m_bh  = ann_metrics(bh_r,  freq=ann_f)

            print("PPO_risk:", m_ppo)
            print("EW      :", m_ew)
            print("BH      :", m_bh)

            rows.append(
                {
                    "Block": "feature",
                    "Freq": freq,
                    "FeatureCfg": cfg,
                    "Reward": "risk",
                    "Model": "PPO",
                    "AnnRet": m_ppo["AnnRet"],
                    "AnnVol": m_ppo["AnnVol"],
                    "Sharpe": m_ppo["Sharpe"],
                    "MaxDD": m_ppo["MaxDD"],
                    "EW_Sharpe": m_ew["Sharpe"],
                    "BH_Sharpe": m_bh["Sharpe"],
                }
            )

            fig_path = os.path.join(OUT_DIR, f"feature_{freq}_{cfg}_risk.png")
            plot_equity(
                {"PPO_risk": cum_s, "EW": ew_cum, "BH": bh_cum},
                fig_path,
            )
            print("Saved plot ->", fig_path)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "feature_ablation_PPO_risk.csv")
    df.to_csv(csv_path, index=False)
    print("\n[Block 1] Feature ablation summary ->", csv_path)


# -------------------- Block 2: reward ablation (RAW vs RISK) -------------------- #

def run_reward_ablation():
    """
    For each freq, BEST_FEATURE_CFG, compare RAW vs RISK reward using PPO.
    """
    rows = []

    for freq in FREQUENCIES:
        print(f"\n===== [Block 2] Reward ablation, freq={freq}, feature={BEST_FEATURE_CFG} =====")

        train_prices = load_split_freq(split="train", freq=freq)
        test_prices = load_split_freq(split="test", freq=freq)
        train_r, test_r = to_log_returns(train_prices), to_log_returns(test_prices)

        ann_f = ann_freq_for(freq)

        train_f = build_features(train_r, WINDOW, config=BEST_FEATURE_CFG)
        test_f  = build_features(test_r,  WINDOW, config=BEST_FEATURE_CFG)

        train_r_aligned = train_r.loc[train_f.index]
        test_r_aligned  = test_r.loc[test_f.index]

        for reward_mode in ["raw", "risk"]:
            lambda_risk = 0.0 if reward_mode == "raw" else 0.02
            vol_window  = 20 if reward_mode == "raw" else 15

            print(f"\n--- reward={reward_mode} ---")

            agent = train_ppo(
                train_r_aligned,
                train_f,
                reward_mode=reward_mode,
                timesteps=TIMESTEPS_REWARD,
                lambda_risk=lambda_risk,
                vol_window=vol_window,
            )

            test_env = make_test_env(
                "ppo",
                test_r_aligned,
                test_f,
                reward_mode=reward_mode,
                lambda_risk=lambda_risk,
                vol_window=vol_window,
            )

            rets = rollout(agent, test_env)
            idx = test_f.index[WINDOW:]
            ret_s = pd.Series(rets, index=idx)
            cum_s = ret_s.cumsum()

            test_slice = test_r_aligned.iloc[WINDOW:]
            ew_r, ew_cum, bh_r, bh_cum = baselines_on_slice(test_slice)

            m_ppo = ann_metrics(ret_s, freq=ann_f)
            m_ew  = ann_metrics(ew_r,  freq=ann_f)
            m_bh  = ann_metrics(bh_r,  freq=ann_f)

            print("PPO:", m_ppo)
            print("EW :", m_ew)
            print("BH :", m_bh)

            rows.append(
                {
                    "Block": "reward",
                    "Freq": freq,
                    "FeatureCfg": BEST_FEATURE_CFG,
                    "Reward": reward_mode,
                    "AnnRet": m_ppo["AnnRet"],
                    "AnnVol": m_ppo["AnnVol"],
                    "Sharpe": m_ppo["Sharpe"],
                    "MaxDD": m_ppo["MaxDD"],
                    "EW_Sharpe": m_ew["Sharpe"],
                    "BH_Sharpe": m_bh["Sharpe"],
                }
            )

            fig_path = os.path.join(OUT_DIR, f"reward_{freq}_{reward_mode}.png")
            plot_equity(
                {"PPO": cum_s, "EW": ew_cum, "BH": bh_cum},
                fig_path,
            )
            print("Saved plot ->", fig_path)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "reward_ablation_PPO.csv")
    df.to_csv(csv_path, index=False)
    print("\n[Block 2] Reward ablation summary ->", csv_path)


# -------------------- Block 3: agent ablation (PPO vs DQN, RISK) -------------------- #

def run_agent_ablation():
    """
    PPO vs DQN, RISK reward, BEST_FEATURE_CFG, all freqs.
    """
    rows = []

    for freq in FREQUENCIES:
        print(f"\n===== [Block 3] Agent ablation, freq={freq}, feature={BEST_FEATURE_CFG}, reward=RISK =====")

        train_prices = load_split_freq(split="train", freq=freq)
        test_prices = load_split_freq(split="test", freq=freq)
        train_r, test_r = to_log_returns(train_prices), to_log_returns(test_prices)

        ann_f = ann_freq_for(freq)

        train_f = build_features(train_r, WINDOW, config=BEST_FEATURE_CFG)
        test_f  = build_features(test_r,  WINDOW, config=BEST_FEATURE_CFG)

        train_r_aligned = train_r.loc[train_f.index]
        test_r_aligned  = test_r.loc[test_f.index]

        lambda_risk = 0.02
        vol_window  = 15

        ppo_agent = train_ppo(
            train_r_aligned,
            train_f,
            reward_mode="risk",
            timesteps=TIMESTEPS_AGENT,
            lambda_risk=lambda_risk,
            vol_window=vol_window,
        )
        dqn_agent = train_dqn(
            train_r_aligned,
            train_f,
            reward_mode="risk",
            timesteps=TIMESTEPS_AGENT,
            lambda_risk=lambda_risk,
            vol_window=vol_window,
        )

        ppo_env = make_test_env("ppo", test_r_aligned, test_f, "risk", lambda_risk, vol_window)
        dqn_env = make_test_env("dqn", test_r_aligned, test_f, "risk", lambda_risk, vol_window)

        ppo_rets = rollout(ppo_agent, ppo_env)
        dqn_rets = rollout(dqn_agent, dqn_env)

        idx = test_f.index[WINDOW:]
        ppo_ret = pd.Series(ppo_rets, index=idx)
        dqn_ret = pd.Series(dqn_rets, index=idx)
        ppo_cum = ppo_ret.cumsum()
        dqn_cum = dqn_ret.cumsum()

        test_slice = test_r_aligned.iloc[WINDOW:]
        ew_r, ew_cum, bh_r, bh_cum = baselines_on_slice(test_slice)

        m_ppo = ann_metrics(ppo_ret, freq=ann_f)
        m_dqn = ann_metrics(dqn_ret, freq=ann_f)
        m_ew  = ann_metrics(ew_r,    freq=ann_f)
        m_bh  = ann_metrics(bh_r,    freq=ann_f)

        print("PPO_risk:", m_ppo)
        print("DQN_risk:", m_dqn)
        print("EW      :", m_ew)
        print("BH      :", m_bh)

        rows.extend(
            [
                {
                    "Block": "agent",
                    "Freq": freq,
                    "FeatureCfg": BEST_FEATURE_CFG,
                    "Reward": "risk",
                    "Model": "PPO",
                    **m_ppo,
                },
                {
                    "Block": "agent",
                    "Freq": freq,
                    "FeatureCfg": BEST_FEATURE_CFG,
                    "Reward": "risk",
                    "Model": "DQN",
                    **m_dqn,
                },
            ]
        )

        fig_path = os.path.join(OUT_DIR, f"agent_{freq}_risk.png")
        plot_equity(
            {"PPO_risk": ppo_cum, "DQN_risk": dqn_cum, "EW": ew_cum, "BH": bh_cum},
            fig_path,
        )
        print("Saved plot ->", fig_path)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "agent_ablation_PPO_vs_DQN_risk.csv")
    df.to_csv(csv_path, index=False)
    print("\n[Block 3] Agent ablation summary ->", csv_path)


# -------------------- main -------------------- #

if __name__ == "__main__":
    run_feature_ablation()
    run_reward_ablation()
    run_agent_ablation()
