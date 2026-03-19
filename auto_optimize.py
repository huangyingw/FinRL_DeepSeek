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
    **kwargs,
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

    pi_optimizer = torch.optim.Adam(actor.parameters(), lr=pi_lr, weight_decay=weight_decay)
    vf_optimizer = torch.optim.Adam(critic.parameters(), lr=vf_lr, weight_decay=weight_decay)

    # 学习率余弦退火
    from torch.optim.lr_scheduler import CosineAnnealingLR
    pi_scheduler = CosineAnnealingLR(pi_optimizer, T_max=epochs, eta_min=pi_lr * 0.01)
    vf_scheduler = CosineAnnealingLR(vf_optimizer, T_max=epochs, eta_min=vf_lr * 0.01)

    # 观测噪声
    obs_noise_std = kwargs.get('obs_noise_std', 0.01)

    # 早停：跟踪最佳验证得分
    best_val_score = -float('inf')
    best_actor_state = None
    best_critic_state = None
    no_improve_count = 0
    early_stop_patience = 10

    # 训练循环
    for epoch in range(epochs):
        obs, _ = train_env.reset()
        obs = np.asarray(obs, dtype=np.float32)
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

            obs = np.asarray(next_obs, dtype=np.float32)
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

        # 每 5 epochs 在验证集上评估
        if epoch % 5 == 0 or epoch == epochs - 1:
            val_obs, _ = val_env.reset()
            val_obs = np.asarray(val_obs, dtype=np.float32)
            val_done = False
            val_portfolio = [initial_amount]

            while not val_done:
                val_obs_t = torch.as_tensor(val_obs, dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    val_dist = actor(val_obs_t)
                    val_action = val_dist.mean.cpu().numpy()
                val_obs_raw, _, val_term, val_trunc, _ = val_env.step(val_action)
                val_obs = np.asarray(val_obs_raw, dtype=np.float32)
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
            val_score = 0.4 * val_sharpe + 0.4 * val_total_return - 0.2 * val_dd

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

    # 复合评分：收益 + 风险调整 - 回撤惩罚
    score = 0.4 * sharpe + 0.4 * total_return - 0.2 * max_dd

    logger.info(f"Trial {trial_name}: Return={total_return*100:.2f}%, Sharpe={sharpe:.3f}, MaxDD={max_dd*100:.1f}%, Score={score:.4f}")

    return score


def objective(trial: Trial, train_df: pd.DataFrame, val_df: pd.DataFrame) -> float:
    """Optuna 目标函数"""
    # 超参数搜索空间（扩展范围，包含正则化参数）
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
