#!/usr/bin/env python
"""
FinRL-DeepSeek 自动化超参数优化
使用 Optuna 进行贝叶斯优化，自动搜索最佳超参数

使用方法:
    python auto_optimize.py --trials 50
    docker compose run finrl-deepseek-optimizer
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import optuna
from optuna.trial import Trial
import torch

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 模型和结果保存目录 - 从配置文件读取
from config_loader import get_models_dir, get_training_config
MODELS_DIR = get_models_dir()
RESULTS_DIR = os.path.join(MODELS_DIR, 'optuna_results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data():
    """加载训练和验证数据"""
    training_config = get_training_config()
    data_source = os.environ.get('DATA_SOURCE', training_config['data_source']).lower()

    if data_source == 'clickhouse':
        logger.info("从 ClickHouse 加载数据...")
        try:
            from clickhouse_data_adapter import load_training_data
            lookback_days = training_config['lookback_days']
            test_ratio = training_config['test_ratio']
            # 优先使用环境变量指定的日期范围
            start_date = os.environ.get('TRAIN_START_DATE')
            end_date = os.environ.get('TRAIN_END_DATE')
            if start_date and end_date:
                train_df, val_df = load_training_data(
                    start_date=start_date, end_date=end_date,
                    test_ratio=test_ratio)
            else:
                train_df, val_df = load_training_data(
                    lookback_days=lookback_days,
                    test_ratio=test_ratio)
            logger.info(f"ClickHouse 数据: 训练 {len(train_df)} 行, 验证 {len(val_df)} 行")
            if len(train_df) > 0:
                logger.info(f"日期范围: {train_df['date'].min()} ~ {val_df['date'].max() if len(val_df) > 0 else train_df['date'].max()}")
            return train_df, val_df
        except Exception as e:
            logger.warning(f"ClickHouse 加载失败: {e}, 回退到 Hugging Face")

    # Hugging Face 数据
    logger.info("从 Hugging Face 加载数据...")
    from datasets import load_dataset
    dataset = load_dataset("benstaf/nasdaq_2013_2023", data_files="train_data_deepseek_risk_2013_2018.csv")
    df = pd.DataFrame(dataset['train'])
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    # 按日期划分
    unique_dates = sorted(df['date'].unique())
    split_idx = int(len(unique_dates) * 0.8)
    split_date = unique_dates[split_idx]

    train_df = df[df['date'] < split_date].reset_index(drop=True)
    val_df = df[df['date'] >= split_date].reset_index(drop=True)

    logger.info(f"HuggingFace 数据: 训练 {len(train_df)} 行, 验证 {len(val_df)} 行")
    return train_df, val_df


def create_env(df, hmax, initial_amount, reward_scaling):
    """创建交易环境"""
    from env_stocktrading_llm_risk import StockTradingEnv

    INDICATORS = [
        'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
        'close_30_sma', 'close_60_sma'
    ]

    # 创建索引
    unique_dates = df['date'].unique()
    date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
    df = df.copy()
    df['new_idx'] = df['date'].map(date_to_idx)
    df = df.set_index('new_idx')

    # 填充缺失值
    df['llm_sentiment'].fillna(0, inplace=True)
    df['llm_risk'].fillna(3, inplace=True)

    stock_dimension = len(df.tic.unique())
    state_space = 1 + 2*stock_dimension + (2+len(INDICATORS))*stock_dimension

    env_kwargs = {
        "hmax": hmax,
        "initial_amount": initial_amount,
        "num_stock_shares": [0] * stock_dimension,
        "buy_cost_pct": [0.001] * stock_dimension,
        "sell_cost_pct": [0.001] * stock_dimension,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": reward_scaling
    }

    env = StockTradingEnv(df=df, **env_kwargs)
    return env


def train_and_evaluate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    epochs: int,
    gamma: float,
    clip_ratio: float,
    pi_lr: float,
    vf_lr: float,
    train_pi_iters: int,
    train_v_iters: int,
    lam: float,
    target_kl: float,
    hmax: int,
    initial_amount: int,
    reward_scaling: float,
    hidden_sizes: tuple,
    trial_name: str,
    ent_coef: float = 0.01,
    weight_decay: float = 1e-5,
    _checkpoint_dir: str = None,
    **kwargs,
) -> float:
    """训练模型并返回验证集收益率

    _checkpoint_dir: 可选，训练完成后把 (验证集早停恢复后的) actor/critic state_dict
    + 训练 config 写入该目录。供 Ray Tune trainable 调 train.report(checkpoint=...) 持久化。
    """
    import warnings
    warnings.filterwarnings('ignore')

    # 创建环境
    train_env = create_env(train_df, hmax, initial_amount, reward_scaling)
    val_env = create_env(val_df, hmax, initial_amount, reward_scaling)

    # 导入 PPO 组件
    import scipy.signal
    from gymnasium.spaces import Box
    import torch.nn as nn
    from torch.distributions.normal import Normal

    import spinup.algos.pytorch.ppo.core as core
    from spinup.utils.logx import EpochLogger

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 简化的 Actor-Critic 网络
    def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
        layers = []
        for j in range(len(sizes)-1):
            act = activation if j < len(sizes)-2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
        return nn.Sequential(*layers)

    class GaussianActor(nn.Module):
        def __init__(self, obs_dim, act_dim, hidden_sizes):
            super().__init__()
            self.net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation=nn.Tanh)
            self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))

        def forward(self, obs):
            mu = self.net(obs)
            std = torch.exp(self.log_std)
            return Normal(mu, std)

        def get_action(self, obs):
            with torch.no_grad():
                dist = self.forward(obs)
                action = dist.sample()
                return action.cpu().numpy()

    class Critic(nn.Module):
        def __init__(self, obs_dim, hidden_sizes):
            super().__init__()
            self.net = mlp([obs_dim] + list(hidden_sizes) + [1], activation=nn.Tanh)

        def forward(self, obs):
            return self.net(obs).squeeze(-1)

    # 创建网络
    obs_dim = train_env.observation_space.shape[0]
    act_dim = train_env.action_space.shape[0]

    actor = GaussianActor(obs_dim, act_dim, hidden_sizes).to(DEVICE)
    critic = Critic(obs_dim, hidden_sizes).to(DEVICE)

    pi_optimizer = torch.optim.Adam(actor.parameters(), lr=pi_lr, weight_decay=weight_decay)
    vf_optimizer = torch.optim.Adam(critic.parameters(), lr=vf_lr, weight_decay=weight_decay)

    # 学习率余弦退火
    from torch.optim.lr_scheduler import CosineAnnealingLR
    pi_scheduler = CosineAnnealingLR(pi_optimizer, T_max=epochs, eta_min=pi_lr * 0.01)
    vf_scheduler = CosineAnnealingLR(vf_optimizer, T_max=epochs, eta_min=vf_lr * 0.01)

    # 观测噪声
    obs_noise_std = kwargs.get('obs_noise_std', 0.01)

    # 观测归一化 Running Mean/Std（与 standalone 训练保持一致，消除评估偏差）
    obs_rms_mean = np.zeros(obs_dim, dtype=np.float64)
    obs_rms_var = np.ones(obs_dim, dtype=np.float64)
    obs_rms_count = 0

    def normalize_obs(obs_raw):
        nonlocal obs_rms_mean, obs_rms_var, obs_rms_count
        obs_flat = np.asarray(obs_raw, dtype=np.float64).flatten()
        obs_flat = np.nan_to_num(obs_flat, nan=0.0, posinf=1e6, neginf=-1e6)
        obs_rms_count += 1
        delta = obs_flat - obs_rms_mean
        obs_rms_mean = obs_rms_mean + delta / obs_rms_count
        delta2 = obs_flat - obs_rms_mean
        obs_rms_var = obs_rms_var + delta * delta2
        if obs_rms_count < 100:
            return np.clip(obs_flat, -10, 10).astype(np.float32).reshape(obs_raw.shape)
        std = np.sqrt(obs_rms_var / obs_rms_count + 1e-8)
        return np.clip((obs_flat - obs_rms_mean) / std, -10, 10).astype(np.float32).reshape(obs_raw.shape)

    # 早停：跟踪最佳验证得分
    best_val_score = -float('inf')
    best_actor_state = None
    best_critic_state = None
    no_improve_count = 0
    early_stop_patience = 10

    # Diagnostic: do-nothing / value-collapse 排查
    # 详见 docs/decisions/2026-05-18-finrl-deepseek-rl-instrumentation.md
    _diag_ev: list = []
    _diag_act_mean_abs: list = []
    _diag_act_zero_frac: list = []

    # 训练循环
    for epoch in range(epochs):
        obs, _ = train_env.reset()
        obs = normalize_obs(np.asarray(obs, dtype=np.float32))
        done = False
        ep_ret = 0

        obs_buf, act_buf, rew_buf, val_buf, logp_buf = [], [], [], [], []

        while not done:
            # 训练时添加观测噪声
            obs_noisy = obs + np.random.normal(0, obs_noise_std, size=obs.shape).astype(np.float32) if obs_noise_std > 0 else obs
            obs_tensor = torch.as_tensor(obs_noisy, dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                dist = actor(obs_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum()
                value = critic(obs_tensor)

            action_np = action.cpu().numpy()
            next_obs, reward, terminated, truncated, _ = train_env.step(action_np)
            done = terminated or truncated

            obs_buf.append(obs)
            act_buf.append(action_np)
            rew_buf.append(reward)
            val_buf.append(value.item())
            logp_buf.append(log_prob.item())

            obs = normalize_obs(np.asarray(next_obs, dtype=np.float32))
            ep_ret += reward

        # 计算优势和回报
        rews = np.array(rew_buf)
        vals = np.array(val_buf)

        # GAE
        deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
        adv = np.zeros_like(rews)
        for t in reversed(range(len(deltas))):
            adv[t] = deltas[t] + gamma * lam * (adv[t+1] if t+1 < len(adv) else 0)

        # 更新网络
        obs_tensor = torch.as_tensor(np.array(obs_buf), dtype=torch.float32, device=DEVICE)
        act_tensor = torch.as_tensor(np.array(act_buf), dtype=torch.float32, device=DEVICE)
        adv_tensor = torch.as_tensor(adv, dtype=torch.float32, device=DEVICE)
        ret_tensor = torch.as_tensor(vals + adv, dtype=torch.float32, device=DEVICE)
        logp_old = torch.as_tensor(logp_buf, dtype=torch.float32, device=DEVICE)

        # Policy 更新
        for _ in range(train_pi_iters):
            dist = actor(obs_tensor)
            logp = dist.log_prob(act_tensor).sum(-1)
            ratio = torch.exp(logp - logp_old)

            clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv_tensor
            loss_pi = -(torch.min(ratio * adv_tensor, clip_adv)).mean()

            # Entropy bonus：鼓励策略保持探索性
            ent = dist.entropy().mean()
            loss_pi = loss_pi - ent_coef * ent

            pi_optimizer.zero_grad()
            loss_pi.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            pi_optimizer.step()

            # Per-dimension KL：除以动作维度数，与 standalone trainer 一致
            kl = (logp_old - logp).mean().item() / act_dim
            if kl > 1.5 * target_kl:
                break

        # Value 更新（带 clipping 防止 loss 爆炸）
        for _ in range(train_v_iters):
            value_pred = critic(obs_tensor)
            loss_v = ((value_pred - ret_tensor)**2).mean()
            loss_v = torch.clamp(loss_v, max=100.0)

            vf_optimizer.zero_grad()
            loss_v.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            vf_optimizer.step()

        # 学习率退火
        pi_scheduler.step()
        vf_scheduler.step()

        # Diagnostic: 每 epoch 末记录 explained_variance + action 分布统计
        with torch.no_grad():
            _final_v = critic(obs_tensor).detach().cpu().numpy()
        _ret_np = ret_tensor.detach().cpu().numpy()
        _ret_var = float(_ret_np.var())
        _residual_var = float((_ret_np - _final_v).var())
        _ev = 1.0 - _residual_var / (_ret_var + 1e-8)
        _act_np = act_tensor.detach().cpu().numpy()
        _diag_ev.append(_ev)
        _diag_act_mean_abs.append(float(np.abs(_act_np).mean()))
        _diag_act_zero_frac.append(float((np.abs(_act_np) < 0.01).mean()))

        # 每 5 epochs 在验证集上评估
        if epoch % 5 == 0 or epoch == epochs - 1:
            val_obs, _ = val_env.reset()
            val_obs = normalize_obs(np.asarray(val_obs, dtype=np.float32))
            val_done = False
            val_portfolio = [initial_amount]

            while not val_done:
                val_obs_t = torch.as_tensor(val_obs, dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    val_dist = actor(val_obs_t)
                    val_action = val_dist.mean.cpu().numpy()
                val_obs_raw, _, val_term, val_trunc, _ = val_env.step(val_action)
                val_obs = normalize_obs(np.asarray(val_obs_raw, dtype=np.float32))
                val_done = val_term or val_trunc
                s = val_env.state
                sd = val_env.stock_dim
                pv = s[0] + sum(s[1+i] * s[1+sd+i] for i in range(sd))
                val_portfolio.append(pv)

            val_pv = pd.Series(val_portfolio)
            val_returns = val_pv.pct_change().dropna()
            val_total_return = (val_portfolio[-1] - initial_amount) / initial_amount
            val_sharpe = val_returns.mean() / val_returns.std() * (252 ** 0.5) if len(val_returns) > 1 and val_returns.std() > 0 else 0.0
            val_cummax = val_pv.cummax()
            val_dd = abs(((val_pv - val_cummax) / val_cummax).min())
            # 加强 MaxDD 惩罚（原 0.2 → 0.4），因完整训练观察到大网络易崩溃到 MaxDD 30%+
            val_score = 0.3 * val_sharpe + 0.3 * val_total_return - 0.4 * val_dd

            if val_score > best_val_score:
                best_val_score = val_score
                best_actor_state = {k: v.clone() for k, v in actor.state_dict().items()}
                best_critic_state = {k: v.clone() for k, v in critic.state_dict().items()}
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= early_stop_patience:
                break

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Train Return = {ep_ret:.2f}")

    # 恢复最佳模型权重进行最终评估
    if best_actor_state is not None:
        actor.load_state_dict(best_actor_state)
        critic.load_state_dict(best_critic_state)

    # Checkpoint 持久化（Ray Tune trainable 触发）—— 写入验证集早停恢复后的 best weights
    # 详见 docs/decisions/2026-05-18-finrl-deepseek-ray-tune-checkpoint-persistence.md
    if _checkpoint_dir is not None:
        import os, json
        os.makedirs(_checkpoint_dir, exist_ok=True)
        torch.save(actor.state_dict(), os.path.join(_checkpoint_dir, 'actor.pt'))
        torch.save(critic.state_dict(), os.path.join(_checkpoint_dir, 'critic.pt'))
        _meta = {
            'trial_name': trial_name,
            'best_val_score': float(best_val_score),
            'epochs_trained': int(epoch + 1),
            'config': {
                'epochs': epochs, 'gamma': gamma, 'clip_ratio': clip_ratio,
                'pi_lr': pi_lr, 'vf_lr': vf_lr,
                'train_pi_iters': train_pi_iters, 'train_v_iters': train_v_iters,
                'lam': lam, 'target_kl': target_kl, 'hmax': hmax,
                'reward_scaling': reward_scaling,
                'hidden_sizes': list(hidden_sizes),
                'ent_coef': ent_coef, 'weight_decay': weight_decay,
            },
        }
        with open(os.path.join(_checkpoint_dir, 'meta.json'), 'w') as _f:
            json.dump(_meta, _f, indent=2)

    # 验证集最终评估
    obs, _ = val_env.reset()
    obs = np.asarray(obs, dtype=np.float32)
    done = False
    portfolio_values = [initial_amount]

    while not done:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
        action = actor.get_action(obs_tensor)
        obs_raw, reward, terminated, truncated, _ = val_env.step(action)
        obs = np.asarray(obs_raw, dtype=np.float32)
        done = terminated or truncated
        s = val_env.state
        sd = val_env.stock_dim
        pv = s[0] + sum(s[1+i] * s[1+sd+i] for i in range(sd))
        portfolio_values.append(pv)

    # 计算收益率
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_amount) / initial_amount

    # 计算 Sharpe 和最大回撤用于复合评分
    pv_series = pd.Series(portfolio_values)
    daily_returns = pv_series.pct_change().dropna()

    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * (252 ** 0.5)
    else:
        sharpe = 0.0

    cummax = pv_series.cummax()
    drawdown = ((pv_series - cummax) / cummax).min()
    max_dd = abs(drawdown) if not pd.isna(drawdown) else 0.0

    # 复合评分：加强 MaxDD 惩罚以过滤掉完整训练会崩溃的参数组合
    score = 0.3 * sharpe + 0.3 * total_return - 0.4 * max_dd

    logger.info(f"Trial {trial_name}: Return={total_return*100:.2f}%, Sharpe={sharpe:.3f}, MaxDD={max_dd*100:.1f}%, Score={score:.4f}")

    # MLflow 记录 trial（HPO 父 run 中的 nested run）
    try:
        from pkg.mlops import log_run
        with log_run(experiment="finrl_deepseek", run_name=trial_name,
                     tags={"entry": "hpo_trial"}, nested=True) as _mlrun:
            _mlrun.log_params({k: v for k, v in kwargs.items()
                               if isinstance(v, (int, float, str, bool))})
            _mlrun.log_params({
                "epochs": epochs, "gamma": gamma, "clip_ratio": clip_ratio,
                "pi_lr": pi_lr, "vf_lr": vf_lr,
                "train_pi_iters": train_pi_iters, "train_v_iters": train_v_iters,
                "lam": lam, "target_kl": target_kl, "hmax": hmax,
                "reward_scaling": reward_scaling,
                "hidden_size_1": hidden_sizes[0], "hidden_size_2": hidden_sizes[1],
                "ent_coef": ent_coef, "weight_decay": weight_decay,
            })
            _mlrun.log_metrics({
                "score": score,
                "total_return": total_return,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
            })
            # Diagnostic 汇总（last 10 epoch 平均，比 final 单点更稳健）
            if _diag_ev:
                _tail = slice(-10, None) if len(_diag_ev) >= 10 else slice(None)
                _mlrun.log_metrics({
                    "diag_ev_tail_avg": float(np.mean(_diag_ev[_tail])),
                    "diag_ev_final": float(_diag_ev[-1]),
                    "diag_act_mean_abs_tail_avg": float(np.mean(_diag_act_mean_abs[_tail])),
                    "diag_act_zero_frac_tail_avg": float(np.mean(_diag_act_zero_frac[_tail])),
                })
    except Exception as _e:
        logger.debug(f"MLflow trial 记录跳过: {_e}")

    return score


def objective(trial: Trial, train_df: pd.DataFrame, val_df: pd.DataFrame) -> float:
    """[已废弃] Optuna 单 fidelity 目标函数

    已被 Ray Tune Multi-Fidelity ASHA 替代（见 main_raytune），保留仅供向后兼容。
    新代码请走 main_raytune 路径，由 ASHA 自动处理 fidelity 维度。
    """
    params = {
        'epochs': trial.suggest_int('epochs', 30, 200),
        'gamma': trial.suggest_float('gamma', 0.95, 0.999),
        'clip_ratio': trial.suggest_float('clip_ratio', 0.1, 0.3),
        'pi_lr': trial.suggest_float('pi_lr', 1e-5, 1e-3, log=True),
        'vf_lr': trial.suggest_float('vf_lr', 1e-5, 1e-3, log=True),
        'train_pi_iters': trial.suggest_int('train_pi_iters', 5, 80),
        'train_v_iters': trial.suggest_int('train_v_iters', 5, 80),
        'lam': trial.suggest_float('lam', 0.9, 0.99),
        'target_kl': trial.suggest_float('target_kl', 0.005, 0.1),
        'hmax': trial.suggest_int('hmax', 50, 300),
        'initial_amount': 1000000,
        'reward_scaling': trial.suggest_float('reward_scaling', 1e-6, 1e-2, log=True),
        'hidden_sizes': (
            trial.suggest_categorical('hidden_size_1', [64, 128, 256]),
            trial.suggest_categorical('hidden_size_2', [64, 128, 256]),
        ),
        'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.05),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'obs_noise_std': trial.suggest_float('obs_noise_std', 0.0, 0.05),
        'trial_name': f"trial_{trial.number}"
    }

    try:
        returns = train_and_evaluate(train_df, val_df, **params)
        return returns
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        return -1.0  # 失败的 trial 返回负收益


# ============================================================
# Ray Tune + Multi-Fidelity ASHA（推荐路径）
#
# 设计理由（见 docs/AUTOML_FRAMEWORK_SELECTION.md）：
# - epoch 作为 fidelity 维度，ASHA 自动决定何时把"小 epoch 表现好的配置"提升到完整训练
# - OptunaSearch Sampler 保留 Bayesian 采样能力 + Ray Tune 的并行/调度
# - 解除 hidden_sizes 等手工硬编码约束（让框架学）
# ============================================================

def raytune_train_and_evaluate(config: dict, train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Ray Tune trainable: 接收 config dict，训练完报告 score 给 ASHA 并持久化 checkpoint"""
    import time
    import tempfile

    # config 包含完整搜索空间 + ASHA 注入的 max_epoch（fidelity）
    params = dict(config)
    # ASHA 用 epochs 作为 fidelity 维度
    epochs = int(params.pop('epochs'))
    hidden_sizes = (int(params.pop('hidden_size_1')), int(params.pop('hidden_size_2')))
    params['hidden_sizes'] = hidden_sizes
    params['epochs'] = epochs
    params['initial_amount'] = 1000000
    params['trial_name'] = f"ray_{int(time.time())}"

    # Ray Tune Checkpoint API（Ray 2.x: ray.train.report + Checkpoint.from_directory）
    # 详见 docs/decisions/2026-05-18-finrl-deepseek-ray-tune-checkpoint-persistence.md
    try:
        from ray import train as _ray_train
        from ray.train import Checkpoint
        _new_api = True
    except ImportError:
        from ray import tune as _ray_train  # type: ignore
        Checkpoint = None  # type: ignore
        _new_api = False

    try:
        with tempfile.TemporaryDirectory() as _ckpt_dir:
            score = train_and_evaluate(train_df, val_df, _checkpoint_dir=_ckpt_dir, **params)
            if _new_api and Checkpoint is not None:
                _ray_train.report({'score': score, 'epochs': epochs},
                                  checkpoint=Checkpoint.from_directory(_ckpt_dir))
            else:
                _ray_train.report({'score': score, 'epochs': epochs})
    except Exception as e:
        logger.error(f"Ray Tune trial failed: {e}")
        if _new_api:
            _ray_train.report({'score': -1.0, 'epochs': epochs})
        else:
            _ray_train.report({'score': -1.0, 'epochs': epochs})


def build_raytune_search_space() -> dict:
    """Ray Tune 搜索空间（不再手工硬编码约束，让 ASHA 自动学习）"""
    from ray import tune
    return {
        'epochs': tune.choice([50, 100, 138, 180, 200]),  # ASHA fidelity 维度
        'gamma': tune.uniform(0.95, 0.999),
        'clip_ratio': tune.uniform(0.1, 0.3),
        'pi_lr': tune.loguniform(1e-5, 1e-3),
        'vf_lr': tune.loguniform(1e-5, 1e-3),
        'train_pi_iters': tune.randint(5, 80),
        'train_v_iters': tune.randint(5, 80),
        'lam': tune.uniform(0.9, 0.99),
        'target_kl': tune.uniform(0.005, 0.1),
        'hmax': tune.randint(50, 300),
        'reward_scaling': tune.loguniform(1e-6, 1e-2),
        'hidden_size_1': tune.choice([64, 128, 256]),
        'hidden_size_2': tune.choice([64, 128, 256]),
        'ent_coef': tune.uniform(0.0, 0.05),
        'weight_decay': tune.loguniform(1e-6, 1e-3),
        'obs_noise_std': tune.uniform(0.0, 0.05),
    }


def raytune_run(train_df, val_df, n_trials: int, max_concurrent: int = 1):
    """Ray Tune + OptunaSearch + ASHA 主入口

    points_to_evaluate 用 Round 7/9a/11 真实回测最佳点作先验热启动
    """
    from functools import partial
    from ray import tune
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.schedulers import ASHAScheduler

    search_space = build_raytune_search_space()

    # 历史完整训练真实最佳作为 prior（Round 11 winner: long_train_180）
    points_to_evaluate = [{
        'epochs': 180,
        'gamma': 0.972709116752409,
        'clip_ratio': 0.11227346352610473,
        'pi_lr': 2.0131428489540617e-05,
        'vf_lr': 0.0005678133375978763,
        'train_pi_iters': 33,
        'train_v_iters': 7,
        'lam': 0.9899690665828681,
        'target_kl': 0.04719432475234017,
        'hmax': 194,
        'reward_scaling': 6.0879465081638936e-05,
        'hidden_size_1': 64,
        'hidden_size_2': 64,
        'ent_coef': 0.021919826902037476,
        'weight_decay': 3.2598665602231373e-05,
        'obs_noise_std': 0.015,
    }]

    optuna_search = OptunaSearch(
        metric='score',
        mode='max',
        points_to_evaluate=points_to_evaluate,
    )

    # ASHA: 把 epochs 作为 fidelity，把表现差的 trial 早停
    asha = ASHAScheduler(
        metric='score',
        mode='max',
        max_t=200,           # 最大 epoch
        grace_period=30,     # 至少跑 30 epoch 才考虑早停
        reduction_factor=3,  # 每轮淘汰 2/3
    )

    # Checkpoint 持久化：写入 PVC（trading-models 挂载点 /app/models），
    # HPO 跑完可从 best_trial.checkpoint.path 提取 best weights。
    # checkpoint_score_attr='score' + keep_checkpoints_num=1 → 每 trial 仅留最优。
    ray_results_dir = os.environ.get('RAY_RESULTS_DIR',
                                     '/app/models/finrl-deepseek/ray_results')
    os.makedirs(ray_results_dir, exist_ok=True)
    trainable = partial(raytune_train_and_evaluate, train_df=train_df, val_df=val_df)
    analysis = tune.run(
        trainable,
        config=search_space,
        num_samples=n_trials,
        search_alg=optuna_search,
        scheduler=asha,
        max_concurrent_trials=max_concurrent,
        local_dir=ray_results_dir,
        keep_checkpoints_num=1,
        checkpoint_score_attr='score',
        verbose=1,
    )
    return analysis


def main():
    parser = argparse.ArgumentParser(description='FinRL-DeepSeek 自动化超参数优化')
    parser.add_argument('--trials', type=int, default=50, help='优化试验次数')
    parser.add_argument('--study-name', type=str, default='finrl-deepseek-optuna', help='Optuna study 名称')
    parser.add_argument('--engine', choices=['raytune', 'optuna'], default='raytune',
                        help='调参引擎：raytune (Ray Tune + ASHA, 推荐) | optuna (单 fidelity, 已废弃)')
    parser.add_argument('--max-concurrent', type=int, default=1,
                        help='Ray Tune 并发 trial 数（单 GPU=1，多 GPU 可调大）')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"FinRL-DeepSeek 自动化超参数优化（engine={args.engine}）")
    logger.info("=" * 60)

    # 检查 GPU
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("未检测到 GPU，使用 CPU 训练")

    # 加载数据
    train_df, val_df = load_data()

    # MLflow 父 run（HPO 整体），各 trial 在 objective() 内嵌套
    try:
        from pkg.mlops import log_run
        _hpo_ctx = log_run(
            experiment="finrl_deepseek",
            run_name=f"hpo_{args.engine}_{args.study_name}",
            tags={"entry": "hpo_main", "engine": args.engine, "trials": str(args.trials)},
        )
    except Exception as _e:
        logger.warning(f"MLflow 不可用，HPO 父 run 跳过: {_e}")
        from contextlib import nullcontext
        _hpo_ctx = nullcontext()

    with _hpo_ctx as _hpo_run:
        if args.engine == 'raytune':
            # Ray Tune + Multi-Fidelity ASHA（推荐路径）
            analysis = raytune_run(train_df, val_df, n_trials=args.trials,
                                   max_concurrent=args.max_concurrent)
            best_trial = analysis.get_best_trial(metric='score', mode='max')
            best_params = dict(best_trial.config)
            best_value = best_trial.last_result['score']

            logger.info("=" * 60)
            logger.info("Ray Tune 优化完成!")
            logger.info(f"最佳得分: {best_value:.4f}")
            logger.info("最佳参数:")
            for key, value in best_params.items():
                logger.info(f"  {key}: {value}")

            # MLflow 记录最佳 metrics + params
            if _hpo_run is not None:
                try:
                    _hpo_run.log_params({k: v for k, v in best_params.items()
                                         if isinstance(v, (int, float, str, bool))})
                    _hpo_run.log_metrics({"best_score": best_value})
                except Exception as _e:
                    logger.debug(f"MLflow 父 run 记录跳过: {_e}")

            # 保存最佳参数到 ParamStore（Redis 后端，统一参数管理）
            # source 字段携带 mlflow run_id 供互引用
            mlflow_source = f"raytune+asha|mlflow:{_hpo_run.run_id}" if _hpo_run is not None else "raytune+asha"
            try:
                from pkg.params import get_store
                store = get_store()
                updates = {
                    f'finrl_deepseek.{k}': v
                    for k, v in best_params.items()
                }
                store.update_batch(updates, source=mlflow_source, performance=best_value)
                logger.info(
                    f"最佳参数已写入 ParamStore (namespace=finrl_deepseek, "
                    f"{len(updates)} 个参数, best_value={best_value:.4f})"
                )
            except ImportError:
                logger.error(
                    "无法导入 pkg.params，请确认父项目 PYTHONPATH 正确（K8s 已通过镜像 build 挂载）"
                )
                raise

            # 提取 best trial checkpoint → 复制到稳定路径 + 上传 MLflow + 写 ParamStore
            # 详见 docs/decisions/2026-05-18-finrl-deepseek-ray-tune-checkpoint-persistence.md
            best_ckpt_obj = getattr(best_trial, 'checkpoint', None)
            if best_ckpt_obj is not None and getattr(best_ckpt_obj, 'path', None):
                import shutil, datetime
                _src = best_ckpt_obj.path
                _ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                _dst_root = os.environ.get(
                    'BEST_TRIAL_CHECKPOINT_DIR',
                    '/app/models/finrl-deepseek/finrl_deepseek/best_trial')
                _dst = os.path.join(_dst_root, f'best_trial_{_ts}')
                os.makedirs(os.path.dirname(_dst), exist_ok=True)
                shutil.copytree(_src, _dst, dirs_exist_ok=True)
                logger.info(f"Best trial checkpoint 复制: {_src} → {_dst}")

                if _hpo_run is not None:
                    _hpo_run.log_artifact(_dst, artifact_path='best_trial')

                try:
                    store.update_batch({
                        'finrl_deepseek.best_trial_checkpoint_path': _dst,
                        'finrl_deepseek.best_trial_score': float(best_value),
                        'finrl_deepseek.best_trial_ts': _ts,
                    }, source=mlflow_source, performance=best_value)
                    logger.info("Best trial checkpoint 路径已写入 ParamStore")
                except Exception as _e:
                    logger.error(f"ParamStore checkpoint 路径写入失败: {_e}")
                    raise
            else:
                logger.warning(
                    "best_trial 无 checkpoint —— Ray Tune checkpoint config 可能未生效。"
                    "请确认 raytune_train_and_evaluate 调 train.report(checkpoint=...)"
                )

            # 保存所有 trial 结果（仅作为审计/调试用途）
            trials_df = analysis.dataframe()
            trials_file = os.path.join(RESULTS_DIR, 'all_trials.csv')
            trials_df.to_csv(trials_file, index=False)
            logger.info(f"所有试验结果已保存到: {trials_file}")
            return

        # ===== 旧路径：Optuna 单 fidelity（已废弃，仅向后兼容）=====
        logger.warning("⚠️  使用旧 Optuna 路径，存在简化训练 vs 完整训练偏差问题。"
                       "推荐 --engine raytune（见 docs/AUTOML_FRAMEWORK_SELECTION.md）")
        # 跨 run 持久化 + warm-start enqueue（参考 auto_retrain.py:379-409）
        # OPTUNA_STORAGE 环境变量启用 sqlite 持久化（K8s manifest 已注入），
        # 让历次 HPO 累积 top-K 配置作为下次起点（warm-start helper 实证 28× time-to-best）
        study_storage = os.environ.get('OPTUNA_STORAGE') or None
        from datetime import datetime as _dt
        study_name_prefix = 'finrl_retrain_'
        study_name = f'{study_name_prefix}{_dt.now().strftime("%Y%m%d_%H%M%S")}' \
            if study_storage else args.study_name
        study = optuna.create_study(
            study_name=study_name,
            storage=study_storage,
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(),
            load_if_exists=True,
        )
        # warm-start: 跨 run 累积 transfer learning（详见 docs/ab_results/warm_start_validation.md）
        if study_storage:
            try:
                from pkg.params.auto_optimizer import AutoOptimizer
                _opt = AutoOptimizer(study_storage=study_storage)
                warm_points = _opt.load_warm_start_points(
                    'finrl_deepseek', top_k=3,
                    study_name_prefix=study_name_prefix,
                )
                for wp in warm_points:
                    study.enqueue_trial(wp, skip_if_exists=True)
                if warm_points:
                    logger.info(
                        f"warm-start: 从历史 study 取 {len(warm_points)} 个 top points 作为先验"
                    )
            except Exception as exc:
                logger.warning(f"warm-start 加载失败 (non-fatal): {exc}")

        study.optimize(
            lambda trial: objective(trial, train_df, val_df),
            n_trials=args.trials,
            show_progress_bar=True
        )

        logger.info("=" * 60)
        logger.info("优化完成!")
        logger.info(f"最佳收益率: {study.best_value * 100:.2f}%")
        logger.info(f"最佳参数:")
        for key, value in study.best_params.items():
            logger.info(f"  {key}: {value}")

        # MLflow 记录最佳 metrics + params
        if _hpo_run is not None:
            try:
                _hpo_run.log_params({k: v for k, v in study.best_params.items()
                                     if isinstance(v, (int, float, str, bool))})
                _hpo_run.log_metrics({"best_value": study.best_value})
            except Exception as _e:
                logger.debug(f"MLflow 父 run 记录跳过: {_e}")

        # 保存最佳参数到 ParamStore（Redis 后端，统一参数管理）
        # 父项目 pkg.params 路径已挂载在 PYTHONPATH 中
        mlflow_source = f"optuna_legacy|mlflow:{_hpo_run.run_id}" if _hpo_run is not None else "optuna_legacy"
        try:
            from pkg.params import get_store
            store = get_store()
            updates = {
                f'finrl_deepseek.{k}': v
                for k, v in study.best_params.items()
            }
            store.update_batch(updates, source=mlflow_source, performance=study.best_value)
            logger.info(
                f"最佳参数已写入 ParamStore (namespace=finrl_deepseek, "
                f"{len(updates)} 个参数, best_value={study.best_value:.4f})"
            )
        except ImportError:
            logger.error(
                "无法导入 pkg.params，请确认父项目 PYTHONPATH 正确（K8s 已通过镜像 build 挂载）"
            )
            raise

        trials_df = study.trials_dataframe()
        trials_file = os.path.join(RESULTS_DIR, 'all_trials.csv')
        trials_df.to_csv(trials_file, index=False)
        logger.info(f"所有试验结果已保存到: {trials_file}")


if __name__ == '__main__':
    main()
