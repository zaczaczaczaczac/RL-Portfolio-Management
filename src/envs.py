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
    观测: 最近k步特征(展平) + 上一时刻权重
    动作: 连续向量 -> softmax -> 投资权重
    奖励: 组合对数收益 - 交易成本 (L1换手率 * cost_bps)
    """
    metadata = {"render.modes": []}

    def __init__(self, returns, features, window=20, cost_bps=20):
        super().__init__()
        # returns/features 都是 DataFrame 对齐后的 .values
        self.returns = returns.values    # [T, N]
        self.features = features.values  # [T, N_feat] (这里=N, 因为用的就是ret)
        self.index = features.index
        self.T, self.N = self.returns.shape
        self.k = window
        self.cost = cost_bps / 1e4

        obs_dim = self.k * self.features.shape[1] + self.N
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-10, high=10,
                                           shape=(self.N,), dtype=np.float32)

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
        return self._obs(), {}

    def step(self, action):
        w_target = _softmax(action)
        turnover = np.abs(w_target - self.w).sum()
        cost = self.cost * turnover

        r_t = float((self.w * self.returns[self.t]).sum())   # 组合log return
        reward = r_t - cost

        self.w = w_target
        self.t += 1
        terminated = self.t >= self.T
        info = {"turnover": turnover, "date": self.index[self.t-1] if self.t-1 < len(self.index) else None}
        return self._obs(), reward, terminated, False, info
