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

# 模型和结果保存目录
MODELS_DIR = os.environ.get('MODELS_DIR', '/app/models')
RESULTS_DIR = os.path.join(MODELS_DIR, 'optuna_results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data():
    """加载训练和验证数据"""
    data_source = os.environ.get('DATA_SOURCE', 'clickhouse').lower()

    if data_source == 'clickhouse':
        logger.info("从 ClickHouse 加载数据...")
        try:
            from clickhouse_data_adapter import load_training_data
            # 使用 lookback_days 控制数据量，默认约5年
            lookback_days = int(os.environ.get('LOOKBACK_DAYS', 365 * 5))
            train_df, val_df = load_training_data(
                lookback_days=lookback_days,
                test_ratio=0.2
            )
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
    trial_name: str
) -> float:
    """训练模型并返回验证集收益率"""
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

    pi_optimizer = torch.optim.Adam(actor.parameters(), lr=pi_lr)
    vf_optimizer = torch.optim.Adam(critic.parameters(), lr=vf_lr)

    # 训练循环
    for epoch in range(epochs):
        obs, _ = train_env.reset()
        done = False
        ep_ret = 0

        obs_buf, act_buf, rew_buf, val_buf, logp_buf = [], [], [], [], []

        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)

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

            obs = next_obs
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

            pi_optimizer.zero_grad()
            loss_pi.backward()
            pi_optimizer.step()

            # KL divergence 检查
            kl = (logp_old - logp).mean().item()
            if kl > 1.5 * target_kl:
                break

        # Value 更新
        for _ in range(train_v_iters):
            value_pred = critic(obs_tensor)
            loss_v = ((value_pred - ret_tensor)**2).mean()

            vf_optimizer.zero_grad()
            loss_v.backward()
            vf_optimizer.step()

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Train Return = {ep_ret:.2f}")

    # 验证集评估
    obs, _ = val_env.reset()
    done = False
    val_ret = 0

    while not done:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
        action = actor.get_action(obs_tensor)
        obs, reward, terminated, truncated, _ = val_env.step(action)
        done = terminated or truncated
        val_ret += reward

    # 计算收益率 (现金 + 持仓价值)
    stock_dim = val_env.stock_dim
    cash = val_env.state[0]
    holdings = val_env.state[1:1+stock_dim]
    prices = val_env.state[1+stock_dim:1+2*stock_dim]
    holdings_value = sum(h * p for h, p in zip(holdings, prices))
    final_value = cash + holdings_value
    returns = (final_value - initial_amount) / initial_amount

    logger.info(f"Trial {trial_name}: Val Return = {returns*100:.2f}%")

    return returns


def objective(trial: Trial, train_df: pd.DataFrame, val_df: pd.DataFrame) -> float:
    """Optuna 目标函数"""
    # 超参数搜索空间（扩展范围以支持更大网络和更长训练）
    params = {
        'epochs': trial.suggest_int('epochs', 50, 150),  # 扩展 epochs 范围
        'gamma': trial.suggest_float('gamma', 0.95, 0.999),
        'clip_ratio': trial.suggest_float('clip_ratio', 0.1, 0.8),  # 扩展 clip_ratio 范围
        'pi_lr': trial.suggest_float('pi_lr', 1e-5, 1e-3, log=True),
        'vf_lr': trial.suggest_float('vf_lr', 1e-5, 1e-3, log=True),
        'train_pi_iters': trial.suggest_int('train_pi_iters', 40, 120),
        'train_v_iters': trial.suggest_int('train_v_iters', 40, 120),
        'lam': trial.suggest_float('lam', 0.9, 0.99),
        'target_kl': trial.suggest_float('target_kl', 0.01, 0.4),  # 扩展 target_kl 范围
        'hmax': trial.suggest_int('hmax', 50, 200),
        'initial_amount': 1000000,
        'reward_scaling': trial.suggest_float('reward_scaling', 1e-5, 1e-3, log=True),
        'hidden_sizes': (
            trial.suggest_categorical('hidden_size_1', [64, 128, 256, 512]),  # 添加 512
            trial.suggest_categorical('hidden_size_2', [64, 128, 256, 512]),  # 添加 512
        ),
        'trial_name': f"trial_{trial.number}"
    }

    try:
        returns = train_and_evaluate(train_df, val_df, **params)
        return returns
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        return -1.0  # 失败的 trial 返回负收益


def main():
    parser = argparse.ArgumentParser(description='FinRL-DeepSeek 自动化超参数优化')
    parser.add_argument('--trials', type=int, default=50, help='优化试验次数')
    parser.add_argument('--study-name', type=str, default='finrl-deepseek-optuna', help='Optuna study 名称')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("FinRL-DeepSeek 自动化超参数优化")
    logger.info("=" * 60)

    # 检查 GPU
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("未检测到 GPU，使用 CPU 训练")

    # 加载数据
    train_df, val_df = load_data()

    # 创建 Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        direction='maximize',  # 最大化收益率
        pruner=optuna.pruners.MedianPruner()
    )

    # 运行优化
    study.optimize(
        lambda trial: objective(trial, train_df, val_df),
        n_trials=args.trials,
        show_progress_bar=True
    )

    # 保存结果
    logger.info("=" * 60)
    logger.info("优化完成!")
    logger.info(f"最佳收益率: {study.best_value * 100:.2f}%")
    logger.info(f"最佳参数:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")

    # 保存最佳参数
    best_params_file = os.path.join(RESULTS_DIR, 'best_params.json')
    with open(best_params_file, 'w') as f:
        json.dump({
            'best_value': study.best_value,
            'best_params': study.best_params,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    logger.info(f"最佳参数已保存到: {best_params_file}")

    # 保存所有试验结果
    trials_df = study.trials_dataframe()
    trials_file = os.path.join(RESULTS_DIR, 'all_trials.csv')
    trials_df.to_csv(trials_file, index=False)
    logger.info(f"所有试验结果已保存到: {trials_file}")


if __name__ == '__main__':
    main()
