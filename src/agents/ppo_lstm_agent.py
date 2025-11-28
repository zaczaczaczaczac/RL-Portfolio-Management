import numpy as np
import pandas as pd
import os
import stable_baselines3 as sb3
from sb3_contrib import RecurrentPPO
from dataclasses import dataclass
from typing import Optional, Callable, Any, Dict, Union
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
import torch

@dataclass
class PPOLSTMHyperparams:
    """
    Hyperparameters for RecurrentPPO (PPO-LSTM).
    """
    policy: str = "MlpLstmPolicy"   # <- LSTM policy
    verbose: int = 0
    learning_rate: Union[float, Callable[[float], float]] = 3e-4
    batch_size: int = 32
    n_steps: int = 2048
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: Union[float, Callable[[float], float]] = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 42
    device: Union[str, Any] = "cpu"


class PPO_LSTM_Agent:
    def __init__(self, env: gym.Env, h_params: PPOLSTMHyperparams = PPOLSTMHyperparams(), model_kwargs: Optional[Dict[str, Any]]=None):
        self.model_name = "PPO_LSTM_Agent"
        self.env = env
        self.h_params = h_params
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        
        if "policy_kwargs" not in self.model_kwargs:
            self.model_kwargs["policy_kwargs"] = dict(
                lstm_hidden_size=128,
                n_lstm_layers=1,
                shared_lstm=True,
                enable_critic_lstm=False
            )
        
        self.model = RecurrentPPO(
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

        self.lstm_states = None
        self.episode_start = np.array([True], dtype=bool)


    def learn(self, timesteps: int, log_interval: int = 1, pbar: bool = False, callback: Optional[BaseCallback]=None):
        self.model.learn(total_timesteps=timesteps, log_interval=log_interval, callback=callback, progress_bar=pbar)
        print(f"{self.model_name} learning completed")
        return self
    
    def reset_lstm_state(self):
        self.lstm_states = None
        self.episode_start = np.array([True], dtype=bool)
    
    def make_action(self, obs: np.ndarray, done: bool=False, deterministic: bool = True):
        self.episode_start = np.array([done], dtype=bool)
        action, self.lstm_states = self.model.predict(obs, state=self.lstm_states, episode_start=self.episode_start, deterministic=deterministic)
        return action
    
    def save_model(self, save_path: str):
        dir_name = os.path.dirname(save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        self.model.save(save_path)
        return None
    
    @classmethod
    def load_model(cls, model_path: str, env: gym.Env, h_params: PPOLSTMHyperparams = PPOLSTMHyperparams(), **kwargs):
        model = cls(model_path, env, config=h_params, **kwargs)
        model.model = RecurrentPPO.load(model_path, env=env, device=h_params.device)
        model.reset_lstm_state()
        return model
