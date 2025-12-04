# src/envs.py
import gymnasium as gym
import numpy as np

def _softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-12)

class PortfolioEnv(gym.Env):
    """
    observation: latest k step characterisitc + last state weighting
    action: continuous vector -> softmax -> adjust weighting
    reward:
        reward_mode = 'raw'  :  reward = r_t - cost
        reward_mode = 'risk' :  reward = r_t - cost - lambda_risk * sigma_recent

        sigma_recent: recent vol_window numbered std 
    """
    metadata = {"render.modes": []}

    def __init__(self, returns, features, window: int = 20, cost_bps: float = 20.0, reward_mode: str = "raw", lambda_risk: float = 0.0, vol_window: int = 20,):
        super().__init__()
        assert reward_mode in ("raw", "risk")
        self.returns_df = returns.loc[features.index]
        self.features_df = features
        self.index = self.features_df.index

        self.returns = self.returns_df.values    # [T, N]
        self.features = self.features_df.values  # [T, F]
        self.T, self.N = self.returns.shape
        self.k = window
        self.cost = cost_bps / 1e4

        # new mode: reward with risk adjusted shown as lambda risk
        self.reward_mode = reward_mode
        self.lambda_risk = lambda_risk
        self.vol_window = vol_window
        self.port_ret_hist = []  # save log retunr history for sigma_recent calculation

        obs_dim = self.k * self.features.shape[1] + self.N
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-10,
            high=10,
            shape=(self.N,),
            dtype=np.float32
        )

        self._rng = np.random.RandomState(0)
        self.reset()

    def _obs(self):
        feat_window = self.features[self.t-self.k:self.t, :].reshape(-1)
        return np.concatenate([feat_window, self.w]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self.t = self.k
        self.w = np.ones(self.N) / self.N
        self.port_ret_hist = []  # <-- clear history each episode
        return self._obs(), {}

    def _compute_risk_penalty(self, r_t: float) -> float:
        """
        Compute sigma based on the recent vol_window portfolio returns and return lambda_risk * sigma.
        This only takes effect when reward_mode == 'risk' and lambda_risk > 0.
        """
        if self.reward_mode != "risk" or self.lambda_risk <= 0:
            return 0.0

        self.port_ret_hist.append(r_t)

        # if the number of log return is not sufficient, pause the penalty 
        if len(self.port_ret_hist) < self.vol_window:
            return 0.0

        recent = np.array(self.port_ret_hist[-self.vol_window:])
        sigma = recent.std()
        return self.lambda_risk * sigma

    def step(self, action):
        w_target = _softmax(action)
        turnover = np.abs(w_target - self.w).sum()
        cost = self.cost * turnover

        r_t = float((self.w * self.returns[self.t]).sum())   # combination log return
        # r_t = float((w_target * self.returns[self.t]).sum()) 

        self.w = w_target
        self.t += 1

         # compute risk adjusted penalty
        risk_penalty = self._compute_risk_penalty(r_t)
        # print(r_t, cost, risk_penalty)

        reward = r_t - cost - risk_penalty
        terminated = self.t >= self.T
        info = {
            "turnover": turnover,
            "date": self.index[self.t - 1] if self.t - 1 < len(self.index) else None,
            "r_raw": r_t,
            "risk_penalty": risk_penalty,
        }
        return self._obs(), reward, terminated, False, info

class DiscretePortfolioEnv(PortfolioEnv):
    """
    Same observation + reward as PortfolioEnv, but with a *discrete* action space
    so that we can use SB3's DQN.

    Action 0: "hold" / no-rebalance (keep previous weights)
    Actions 1..K: fixed portfolio weight vectors in self.action_weights[i-1]
    """

    def __init__(
        self,
        returns,
        features,
        window: int = 20,
        cost_bps: float = 20.0,
        reward_mode: str = "raw",
        lambda_risk: float = 0.0,
        vol_window: int = 20,
        action_weights: np.ndarray | None = None,
    ):
        super().__init__(
            returns=returns,
            features=features,
            window=window,
            cost_bps=cost_bps,
            reward_mode=reward_mode,
            lambda_risk=lambda_risk,
            vol_window=vol_window,
        )

        if action_weights is None:
            self.action_weights = self._build_default_actions(self.N)
        else:
            w = np.asarray(action_weights, dtype=float)
            assert w.ndim == 2 and w.shape[1] == self.N, \
                "action_weights must be [K, N_assets]"
            # normalize each row to sum to 1
            w = w / (w.sum(axis=1, keepdims=True) + 1e-12)
            self.action_weights = w

        # +1 for the "hold" action at index 0
        self.n_actions = self.action_weights.shape[0] + 1
        self.action_space = gym.spaces.Discrete(self.n_actions)

    @staticmethod
    def _build_default_actions(n_assets: int) -> np.ndarray:
        """
        Build a richer, more realistic action set on a 25% grid:
        - equal-weight over all assets
        - 100% in each single asset
        - 2-asset combinations with weights:
          (0.25, 0.75), (0.5, 0.5), (0.75, 0.25)
        """
        actions = []

        # 1) equal-weight portfolio
        actions.append((np.ones(n_assets, dtype=float) / n_assets).tolist())

        # 2) 100% in each single asset
        eye = np.eye(n_assets, dtype=float)
        actions.extend(eye.tolist())

        # 3) pair portfolios on 25% grid
        splits = [(0.25, 0.75), (0.5, 0.5), (0.75, 0.25)]
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                for w_i, w_j in splits:
                    w = np.zeros(n_assets, dtype=float)
                    w[i] = w_i
                    w[j] = w_j
                    actions.append(w.tolist())

        return np.array(actions, dtype=float)

    def step(self, action):
        action = int(action)

        if action == 0:
            # "hold" action: keep previous weights, no rebalance
            w_target = self.w.copy()
        else:
            # map discrete index -> portfolio weights
            w_target = self.action_weights[action - 1]

        turnover = np.abs(w_target - self.w).sum()
        cost = self.cost * turnover

        # portfolio log return using previous weights
        r_t = float((self.w * self.returns[self.t]).sum())

        # update weights & time
        self.w = w_target
        self.t += 1

        # reward logic identical to PortfolioEnv
        if self.reward_mode == "raw":
            risk_penalty = 0.0
        elif self.reward_mode == "risk":
            risk_penalty = self._compute_risk_penalty(r_t)
        else:
            raise ValueError(f"Unknown reward_mode {self.reward_mode}")

        reward = r_t - cost - risk_penalty
        terminated = self.t >= self.T

        info = {
            "turnover": turnover,
            "date": self.index[self.t - 1] if self.t - 1 < len(self.index) else None,
            "r_raw": r_t,
            "risk_penalty": risk_penalty,
            "action_index": action,
        }
        return self._obs(), reward, terminated, False, info