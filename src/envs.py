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
    奖励:
        reward_mode = 'raw'  :  reward = r_t - cost
        reward_mode = 'risk' :  reward = r_t - cost - lambda_risk * sigma_recent

        其中 sigma_recent 为最近 vol_window 个组合收益的标准差
    """
    metadata = {"render.modes": []}

    def __init__(self, returns, features, window: int = 20, cost_bps: float = 20.0, reward_mode: str = "raw", lambda_risk: float = 0.0, vol_window: int = 20,):
        super().__init__()
        assert reward_mode in ("raw", "risk")
        # 保存 DataFrame 版本，方便以后debug
        self.returns_df = returns.loc[features.index]
        self.features_df = features
        self.index = self.features_df.index

        # numpy 版本用于加速
        self.returns = self.returns_df.values    # [T, N]
        self.features = self.features_df.values  # [T, F]
        self.T, self.N = self.returns.shape
        self.k = window
        self.cost = cost_bps / 1e4

        # 新属性：reward 模式 + 风险惩罚参数
        self.reward_mode = reward_mode
        self.lambda_risk = lambda_risk
        self.vol_window = vol_window
        self.port_ret_hist = []  # 存历史组合log return，用来算 sigma

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
        return self._obs(), {}

    def _compute_risk_penalty(self, r_t: float) -> float:
        """
        基于最近 vol_window 个组合收益计算 sigma，并返回 lambda_risk * sigma
        只在 reward_mode == 'risk' 且 lambda_risk > 0 时生效
        """
        if self.reward_mode != "risk" or self.lambda_risk <= 0:
            return 0.0

        # 记录本期收益
        self.port_ret_hist.append(r_t)

        # 长度不够就先不惩罚
        if len(self.port_ret_hist) < self.vol_window:
            return 0.0

        recent = np.array(self.port_ret_hist[-self.vol_window:])
        sigma = recent.std()
        return self.lambda_risk * sigma

    def step(self, action):
        w_target = _softmax(action)
        turnover = np.abs(w_target - self.w).sum()
        cost = self.cost * turnover

        r_t = float((self.w * self.returns[self.t]).sum())   # 组合log return
        # 计算风险惩罚
        risk_penalty = self._compute_risk_penalty(r_t)

        reward = r_t - cost - risk_penalty

        self.w = w_target
        self.t += 1
        terminated = self.t >= self.T
        info = {
            "turnover": turnover,
            "date": self.index[self.t - 1] if self.t - 1 < len(self.index) else None,
            "r_raw": r_t,
            "risk_penalty": risk_penalty,
        }
        return self._obs(), reward, terminated, False, info