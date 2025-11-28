import os
from dataclasses import dataclass
from typing import Optional, Callable, Any, Dict, Union

import numpy as np
import stable_baselines3 as sb3
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback


@dataclass
class DQNHyperparams:
    """
    DQN Hyperparameter class
    References: https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
    """
    policy: str = "MlpPolicy"
    verbose: int = 0
    learning_rate: Union[float, Callable[[float], float]] = 1e-3
    buffer_size: int = 50_000
    learning_starts: int = 1_000
    batch_size: int = 64
    tau: float = 1.0
    gamma: float = 0.99
    train_freq: int | tuple[int, str] = 1          # can also be (1, "step")
    gradient_steps: int = 1
    target_update_interval: int = 1_000
    exploration_fraction: float = 0.1
    exploration_final_eps: float = 0.05
    seed: int = 42
    device: Union[str, Any] = "cpu"


class DQN_Agent:
    def __init__(
        self,
        env: gym.Env,
        h_params: DQNHyperparams = DQNHyperparams(),
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = "DQN_Agent"
        self.env = env
        self.h_params = h_params
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}

        self.model = sb3.DQN(
            policy=h_params.policy,
            env=env,
            learning_rate=h_params.learning_rate,
            buffer_size=h_params.buffer_size,
            learning_starts=h_params.learning_starts,
            batch_size=h_params.batch_size,
            tau=h_params.tau,
            gamma=h_params.gamma,
            train_freq=h_params.train_freq,
            gradient_steps=h_params.gradient_steps,
            target_update_interval=h_params.target_update_interval,
            exploration_fraction=h_params.exploration_fraction,
            exploration_final_eps=h_params.exploration_final_eps,
            verbose=h_params.verbose,
            seed=h_params.seed,
            device=h_params.device,
            **self.model_kwargs,
        )

    def learn(
        self,
        timesteps: int,
        log_interval: int = 1,
        pbar: bool = False,
        callback: Optional[BaseCallback] = None,
    ):
        self.model.learn(
            total_timesteps=timesteps,
            log_interval=log_interval,
            callback=callback,
            progress_bar=pbar,
        )
        print(f"{self.model_name} learning completed")
        return self

    def make_action(self, obs: np.ndarray, deterministic: bool = True):
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def save_model(self, save_path: str):
        dir_name = os.path.dirname(save_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        self.model.save(save_path)
        return None

    @classmethod
    def load_model(
        cls,
        model_path: str,
        env: gym.Env,
        h_params: DQNHyperparams = DQNHyperparams(),
        **kwargs,
    ):
        agent = cls(env=env, h_params=h_params, model_kwargs=kwargs)
        agent.model = sb3.DQN.load(model_path, env=env, device=h_params.device)
        return agent
