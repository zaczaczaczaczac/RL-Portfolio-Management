import numpy as np
import pandas as pd
import os
import stable_baselines3 as sb3
from dataclasses import dataclass
from typing import Optional, Callable, Any, Dict, Union
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback

@dataclass
class PPOHyperparams():
    """
    PPO Hyperparameter class
    References: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    """
    policy: str = "MlpPolicy"
    verbose: int = 0 
    learning_rate: Union[float, Callable[[float], float]] = 3e-4
    batch_size: int = 64
    n_steps: int = 2048
    n_epochs: int = 15
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: Union[float, Callable[[float], float]] = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 52
    device: Union[str, Any] = "auto"


class PPO_Agent:
    def __init__(self, env: gym.Env, h_params: PPOHyperparams = PPOHyperparams(), model_kwargs: Optional[Dict[str, Any]]=None):
        self.model_name = "PPO_Agent"
        self.env = env
        self.h_params = h_params
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.model = sb3.PPO(
            policy=h_params.policy,
            env=env,
            learning_rate=h_params.learning_rate,
            batch_size=h_params.batch_size,
            n_steps=h_params.n_steps,
            n_epochs=h_params.n_epochs,
            gamma=h_params.gamma,
            gae_lambda=h_params.gae_lambda,
            clip_range=h_params.clip_range,
            ent_coef=h_params.ent_coef,
            vf_coef=h_params.vf_coef,
            max_grad_norm=h_params.max_grad_norm,
            verbose=h_params.verbose,
            seed=h_params.seed,
            device=h_params.device,
            **self.model_kwargs
        )

    def learn(self, timesteps: int, log_interval: int = 1, pbar: bool = False, callback: Optional[BaseCallback]=None):
        self.model.learn(total_timesteps=timesteps, log_interval=log_interval, callback=callback, progress_bar=pbar)
        print(f"{self.model_name} learning completed")
        return self
    
    def make_action(self, obs: np.darray, deterministic: bool = True):
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action
    
    def save_model(self, save_path: str):
        dir_name = os.path.dirname(save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        self.model.save(save_path)
        return None
    
    @classmethod
    def load_model(cls, model_path: str, env: gym.Env, h_params: PPOHyperparams = PPOHyperparams(), **kwargs):
        model = cls(model_path, env, config=h_params, **kwargs)
        model.model = sb3.PPO.load(model_path, env=env, device=h_params.device)
        return model