#!/usr/bin/env python
# coding: utf-8
# Standalone training script for CPPO-DeepSeek
# Run with: OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 mpirun -np 4 python3 train_cppo_llm_risk_standalone.py
#
# 配置来源: config/settings.yaml (Docker 挂载)
#   training.data_source: 'clickhouse' 或 'huggingface' (默认: clickhouse)
#   training.lookback_days: 训练数据回溯天数 (默认: 1825，约5年)
#   training.epochs: 训练轮数 (默认: 100)

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径以导入 MessageBus
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 消息总线支持（可选）
MESSAGE_BUS = None
CORRELATION_ID = None
try:
    from trading_strategies.message_bus import MessageBus, EventType
    MESSAGE_BUS_ENABLED = os.environ.get('ENABLE_MESSAGE_BUS', 'false').lower() == 'true'
except ImportError:
    MESSAGE_BUS_ENABLED = False

from datasets import load_dataset
import pandas as pd
from env_stocktrading_llm_risk import StockTradingEnv

# Define INDICATORS directly (avoiding finrl package dependencies)
INDICATORS = [
    'macd',
    'boll_ub',
    'boll_lb',
    'rsi_30',
    'cci_30',
    'dx_30',
    'close_30_sma',
    'close_60_sma',
]

# Stateless: 模型保存到 Docker named volume
from config_loader import get_models_dir, get_training_config
TRAINED_MODEL_DIR = os.path.join(get_models_dir(), 'finrl_deepseek', 'trained_models')
os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)
print(f"Model directory: {TRAINED_MODEL_DIR}")

import numpy as np
import scipy.signal
from gymnasium.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

import spinup.algos.pytorch.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

import time
import uuid
from datetime import datetime


def publish_event(event_type: str, **kwargs):
    """发布事件到消息总线（如果启用）"""
    global MESSAGE_BUS, CORRELATION_ID
    if MESSAGE_BUS is None:
        return
    try:
        MESSAGE_BUS.publish_training_event(
            event_type,
            correlation_id=CORRELATION_ID,
            **kwargs
        )
    except Exception as e:
        print(f"[MessageBus] Failed to publish {event_type}: {e}")


# Force GPU usage
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This training requires GPU.")
DEVICE = torch.device("cuda")
print(f"Using device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0)}")


def load_data():
    """根据配置文件加载训练和验证数据"""
    training_config = get_training_config()
    data_source = training_config['data_source'].lower()
    test_ratio = float(training_config.get('test_ratio', 0.2))
    val_df = None

    if data_source == 'clickhouse':
        print("Loading training data from ClickHouse...")
        try:
            from clickhouse_data_adapter import load_training_data
            # 优先使用环境变量指定的日期范围，回退到 lookback_days
            start_date = os.environ.get('TRAIN_START_DATE')
            end_date = os.environ.get('TRAIN_END_DATE')
            lookback_days = training_config['lookback_days']
            if start_date and end_date:
                train, val_df = load_training_data(
                    start_date=start_date, end_date=end_date, test_ratio=test_ratio)
            else:
                train, val_df = load_training_data(
                    lookback_days=lookback_days, test_ratio=test_ratio)
            actual_start = train['date'].min()
            actual_end = train['date'].max()
            print(f"Loaded {len(train)} training rows from ClickHouse ({actual_start} to {actual_end})")
            if val_df is not None and len(val_df) > 0:
                print(f"Loaded {len(val_df)} validation rows ({val_df['date'].min()} to {val_df['date'].max()})")
        except Exception as e:
            print(f"ClickHouse 加载失败: {e}")
            print("回退到 Hugging Face 数据...")
            data_source = 'huggingface'

    if data_source == 'huggingface':
        print("Loading training data from Hugging Face...")
        from datasets import load_dataset
        dataset = load_dataset("benstaf/nasdaq_2013_2023", data_files="train_data_deepseek_risk_2013_2018.csv")
        train = pd.DataFrame(dataset['train'])
        if 'Unnamed: 0' in train.columns:
            train = train.drop('Unnamed: 0', axis=1)

        # 按时间拆分验证集
        if test_ratio > 0:
            unique_dates = sorted(train['date'].unique())
            split_idx = int(len(unique_dates) * (1 - test_ratio))
            train_dates = unique_dates[:split_idx]
            val_dates = unique_dates[split_idx:]
            val_df = train[train['date'].isin(val_dates)].copy()
            train = train[train['date'].isin(train_dates)].copy()
            print(f"Split: {len(train)} training rows, {len(val_df)} validation rows")

    return train, val_df


# Load data
train, val_df = load_data()

# Create a new index based on unique dates
unique_dates = train['date'].unique()
date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
train['new_idx'] = train['date'].map(date_to_idx)
train = train.set_index('new_idx')

# Fill missing values
train['llm_sentiment'].fillna(0, inplace=True)  # 0 is outside scope of sentiment scores (min is 1)
train['llm_risk'].fillna(3, inplace=True)  # neutral risk score is 3

# Environment setup
stock_dimension = len(train.tic.unique())
state_space = 1 + 2*stock_dimension + (2+len(INDICATORS))*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

# 保存参考股票列表（供 backtester 使用）
import json as _json
_ref_stocks_path = os.path.join(os.environ.get('MODELS_DIR', '/app/models'), 'reference_stocks.json')
_ref_stocks = sorted(train['tic'].unique().tolist())
with open(_ref_stocks_path, 'w') as _f:
    _json.dump({'stocks': _ref_stocks, 'count': len(_ref_stocks),
                'data_source': os.environ.get('DATA_SOURCE', 'huggingface')}, _f, indent=2)
print(f"Saved reference stocks ({len(_ref_stocks)}) to {_ref_stocks_path}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

# 超参数：自动从 best_params.json 读取（Optuna 优化结果）
def load_best_params():
    """从 Optuna 优化结果自动加载最佳超参数"""
    import json
    best_params_path = os.path.join(
        get_models_dir(),
        'optuna_results', 'best_params.json'
    )
    if os.path.exists(best_params_path):
        with open(best_params_path, 'r') as f:
            data = json.load(f)
            print(f"✅ Loaded best params from {best_params_path}")
            return data.get('best_params', {})
    print("⚠️ best_params.json not found, using defaults")
    return {}

BEST_PARAMS = load_best_params()
# 只从 best_params.json 读取，不使用环境变量回退
HMAX = int(BEST_PARAMS.get('hmax', 100))
REWARD_SCALING = float(BEST_PARAMS.get('reward_scaling', 1e-4))

env_kwargs = {
    "hmax": HMAX,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": REWARD_SCALING
}

# 使用原始 Gymnasium 环境（不使用 SB3 DummyVecEnv 包装器）
# DummyVecEnv 添加 batch 维度导致 CUDA batch vs 单样本计算精度差异，
# 使得 KL 在未更新策略时就已达到 ~0.02（应为 ~0）
env_train = StockTradingEnv(df=train, **env_kwargs)

# 验证环境（用于早停和过拟合检测）
env_val = None
if val_df is not None and len(val_df) > 0:
    # 验证集需要相同的索引处理
    val_unique_dates = val_df['date'].unique()
    val_date_to_idx = {date: idx for idx, date in enumerate(sorted(val_unique_dates))}
    val_df_indexed = val_df.copy()
    val_df_indexed['new_idx'] = val_df_indexed['date'].map(val_date_to_idx)
    val_df_indexed = val_df_indexed.set_index('new_idx')
    val_df_indexed['llm_sentiment'].fillna(0, inplace=True)
    val_df_indexed['llm_risk'].fillna(3, inplace=True)
    env_val = StockTradingEnv(df=val_df_indexed, **env_kwargs)
    print(f"Validation env created: {len(val_unique_dates)} trading days")


# Neural Network Definitions
def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            # 强制使用 2D tensor，确保与 batch update 使用相同的 CUDA GEMM kernel
            # 避免 1D (gemv) vs 2D (gemm) 精度差异导致 logp 不一致
            was_1d = obs.dim() == 1
            if was_1d:
                obs = obs.unsqueeze(0)
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
            if was_1d:
                a = a.squeeze(0)
                logp_a = logp_a.squeeze(0)
                v = v.squeeze(0)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]


# CPPO Buffer
class CPPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.valupdate_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, valupdate, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = float(rew)
        self.val_buf[self.ptr] = float(val)
        self.valupdate_buf[self.ptr] = float(valupdate)
        self.logp_buf[self.ptr] = float(logp)
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        # 只对当前 path 的 advantage 减去 CVaR 更新（修复：之前对整个 buffer 操作导致污染其他 trajectory）
        self.adv_buf[path_slice] = self.adv_buf[path_slice] - self.valupdate_buf[path_slice]
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32, device=DEVICE) for k, v in data.items()}


# CPPO Algorithm - 自动从 best_params.json 读取超参数
def cppo(env_fn,
         actor_critic=core.MLPActorCritic,
         ac_kwargs=None,
         seed=42,
         steps_per_epoch=20000,
         epochs=None,
         gamma=None,
         env_val=None,
         clip_ratio=None,
         pi_lr=None,
         vf_lr=None,
         train_pi_iters=None,
         train_v_iters=None,
         lam=None,
         max_ep_len=3000,
         target_kl=None,
         logger_kwargs=dict(),
         save_freq=10,
         alpha=0.85,
         beta=3000.0,
         nu_lr=5e-4,
         lam_lr=5e-4,
         nu_start=0.1,
         lam_start=0.01,
         nu_delay=0.75,
         lam_low_bound=0.001,
         delay=1.0,
         cvar_clip_ratio=0.05):
    # 只从 best_params.json 读取超参数（Optuna 优化结果）
    # 默认值基于 PPO 文献标准范围
    epochs = epochs or int(BEST_PARAMS.get('epochs', 100))
    gamma = gamma or float(BEST_PARAMS.get('gamma', 0.99))
    clip_ratio = clip_ratio or float(BEST_PARAMS.get('clip_ratio', 0.2))
    pi_lr = pi_lr or float(BEST_PARAMS.get('pi_lr', 3e-4))
    vf_lr = vf_lr or float(BEST_PARAMS.get('vf_lr', 1e-3))
    train_pi_iters = train_pi_iters or int(BEST_PARAMS.get('train_pi_iters', 10))
    train_v_iters = train_v_iters or int(BEST_PARAMS.get('train_v_iters', 10))
    lam = lam or float(BEST_PARAMS.get('lam', 0.95))
    target_kl = target_kl or float(BEST_PARAMS.get('target_kl', 0.03))

    # 网络结构：从 best_params.json 读取（默认 256x128）
    hidden_size_1 = int(BEST_PARAMS.get('hidden_size_1', 256))
    hidden_size_2 = int(BEST_PARAMS.get('hidden_size_2', 128))
    # 使用 Tanh 激活函数（与 auto_optimize.py 一致）
    # ReLU 在未归一化观测值（如 cash=1,000,000）下导致网络输出极大值，
    # 引起 float32 精度问题使 batch 与单样本计算的 logp 不一致（KL 爆炸）
    ac_kwargs = ac_kwargs or dict(hidden_sizes=[hidden_size_1, hidden_size_2], activation=torch.nn.Tanh)

    print(f"Training with optimized hyperparameters:")
    print(f"  epochs={epochs}, gamma={gamma:.4f}, clip_ratio={clip_ratio:.4f}")
    print(f"  pi_lr={pi_lr:.6f}, vf_lr={vf_lr:.6f}")
    print(f"  train_pi_iters={train_pi_iters}, train_v_iters={train_v_iters}")
    print(f"  lam={lam:.4f}, target_kl={target_kl:.4f}")
    print(f"  hidden_sizes=[{hidden_size_1}, {hidden_size_2}]")

    setup_pytorch_for_mpi()

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac = ac.to(DEVICE)  # Move model to GPU
    sync_params(ac)

    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = CPPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    nu = nu_start
    cvarlam = lam_start

    from torch.optim import Adam

    # KL 归一化：对多维动作空间，KL 除以动作维度数
    # logp 是对所有维度求和，所以 KL 也是所有维度之和
    # 归一化为 per-dimension KL 使 target_kl 语义与单维度动作一致
    n_act_dims = act_dim[0] if isinstance(act_dim, tuple) else act_dim

    # Entropy bonus 系数：鼓励探索，防止策略过早收敛
    ent_coef = float(BEST_PARAMS.get('ent_coef', 0.01))
    print(f"  ent_coef={ent_coef}, weight_decay=1e-5")

    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Entropy bonus：鼓励策略保持探索性
        ent = pi.entropy().mean()
        loss_pi = loss_pi - ent_coef * ent

        # Per-dimension KL：除以动作维度数，消除维度数量对 KL 总和的影响
        approx_kl = (logp_old - logp).mean().item() / n_act_dims
        ent_val = ent.item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent_val, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    weight_decay = float(BEST_PARAMS.get('weight_decay', 1e-5))
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr, weight_decay=weight_decay)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr, weight_decay=weight_decay)

    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break
            loss_pi.backward()
            torch.nn.utils.clip_grad_norm_(ac.pi.parameters(), 0.5)
            mpi_avg_grads(ac.pi)
            pi_optimizer.step()

        logger.store(StopIter=i)

        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            # Value loss clipping 防止爆炸
            loss_v = torch.clamp(loss_v, max=100.0)
            loss_v.backward()
            torch.nn.utils.clip_grad_norm_(ac.v.parameters(), 0.5)
            mpi_avg_grads(ac.v)
            vf_optimizer.step()

        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    start_time = time.time()
    # Gymnasium 接口: reset() 返回 (obs, info)
    o_raw, _ = env.reset()
    # 关键：统一 float32 转换路径，确保收集和更新时使用完全相同的数值
    o = np.asarray(o_raw, dtype=np.float32)
    ep_ret, ep_len = 0, 0

    # Early stopping: 连续 patience 次验证不改进则停止训练
    patience = int(BEST_PARAMS.get('early_stopping_patience', 10))
    best_val_return = -float('inf')
    no_improve_count = 0
    early_stopped = False

    for epoch in range(epochs):
        trajectory_num = 0
        bad_trajectory_num = 0
        cvarlam = cvarlam + lam_lr * (beta - nu)
        lam_delta = 0
        nu_delta = 0
        update_num = 0

        for t in range(local_steps_per_epoch):
            # 使用 float32 numpy 数组创建 tensor（避免 Python list → torch 的转换路径差异）
            a, v, logp = ac.step(torch.as_tensor(o, device=DEVICE))

            # Gymnasium 接口: step() 返回 5-tuple
            next_o_raw, r, terminated, truncated, _ = env.step(a)
            d = terminated or truncated
            ep_ret += r
            ep_len += 1

            # 统一转换为 float32 numpy
            next_o = np.asarray(next_o_raw, dtype=np.float32)

            # 从 env.state 读取原始值（观测已归一化，CVaR 计算需要原始值）
            raw_state = env.state
            llm_risks = np.array(raw_state[-stock_dimension:], dtype=np.float32)

            risk_to_weight = {1: 0.99, 2: 0.995, 3: 1.0, 4: 1.005, 5: 1.01}
            llm_risks_weights = np.vectorize(risk_to_weight.get)(llm_risks)

            prices = np.array(raw_state[1:stock_dimension+1], dtype=np.float32)
            shares = np.array(raw_state[stock_dimension+1:stock_dimension*2+1], dtype=np.float32)

            stock_values = prices * shares
            total_value = np.sum(stock_values)
            if total_value == 0:
                llm_risk_factor = 1
            else:
                stock_weights = stock_values / total_value
                llm_risk_factor = np.dot(stock_weights, llm_risks_weights)

            adjusted_D_pi = llm_risk_factor * (ep_ret + v - r)
            trajectory_num += 1
            nu_delta += adjusted_D_pi
            updates = np.float32(0.0)
            if adjusted_D_pi < nu:
                bad_trajectory_num += 1
                lam_delta += adjusted_D_pi
                updates = delay * cvarlam / (1 - alpha) * (nu - adjusted_D_pi)
                if updates > abs(v) * cvar_clip_ratio:
                    updates = abs(v) * cvar_clip_ratio
                    update_num += 1
                updates = np.float32(updates)

            buf.store(o, a, r, v, updates, logp)
            logger.store(VVals=v)

            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, device=DEVICE))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o_raw, _ = env.reset()
                o = np.asarray(o_raw, dtype=np.float32)
                ep_ret, ep_len = 0, 0

        if bad_trajectory_num > 0:
            lam_delta = lam_delta / bad_trajectory_num
        if trajectory_num > 0:
            nu_delta = nu_delta / trajectory_num
        nu = nu_delta * nu_delay

        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, None)

        update()

        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()

        # 验证集评估（每 5 epochs）
        if env_val is not None and (epoch % 5 == 0 or epoch == epochs - 1):
            val_obs, _ = env_val.reset()
            val_done = False
            val_ret = 0
            while not val_done:
                val_obs_t = torch.as_tensor(np.asarray(val_obs, dtype=np.float32), device=DEVICE)
                with torch.no_grad():
                    val_a = ac.pi._distribution(val_obs_t.unsqueeze(0)).mean.squeeze(0)
                val_obs, val_r, val_term, val_trunc, _ = env_val.step(val_a.cpu().numpy())
                val_done = val_term or val_trunc
                val_ret += val_r
            # 计算验证集收益率
            val_cash = env_val.state[0]
            val_holdings = env_val.state[1:1+stock_dimension]
            val_prices = env_val.state[1+stock_dimension:1+2*stock_dimension]
            val_value = val_cash + sum(h * p for h, p in zip(val_holdings, val_prices))
            val_return_pct = (val_value - 1000000) / 1000000 * 100
            print(f"  [Val] Epoch {epoch}: Return={val_return_pct:.2f}%, FinalValue={val_value:.0f}")

            # 保存最佳模型 + 早停检测
            if val_return_pct > best_val_return:
                best_val_return = val_return_pct
                no_improve_count = 0
                best_path = TRAINED_MODEL_DIR + "/best_model.pth"
                torch.save(ac.state_dict(), best_path)
                print(f"  [Val] New best! Saved to {best_path}")
            else:
                no_improve_count += 1
                print(f"  [Val] No improvement for {no_improve_count}/{patience} evaluations")

            if no_improve_count >= patience:
                print(f"\n  [Early Stop] Stopping at epoch {epoch}/{epochs}")
                print(f"  [Early Stop] Best validation return: {best_val_return:.2f}%")
                early_stopped = True
                break

        print("-" * 37)
        print("bad_trajectory_num:", bad_trajectory_num)
        print("update num:", update_num)
        print("nu:", nu)
        print("lam:", cvarlam)
        print("-" * 37, flush=True)

        # 每 10 epochs 发布进度事件
        if MESSAGE_BUS_ENABLED and (epoch + 1) % 10 == 0:
            publish_event(
                EventType.TRAINING_PROGRESS,
                metrics={
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "progress_pct": round((epoch + 1) / epochs * 100, 1),
                    "elapsed_time": time.time() - start_time,
                    "nu": float(nu),
                    "lam": float(cvarlam)
                }
            )
    if early_stopped:
        print(f"Training ended early at epoch {epoch}/{epochs} (best val: {best_val_return:.2f}%)")
    else:
        print(f"Training completed all {epochs} epochs (best val: {best_val_return:.2f}%)")
    return ac


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='env_train')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--exp_name', type=str, default='cppo_deepseek')
    parser.add_argument('-f', '--file', type=str, help='Kernel connection file')
    parser.add_argument('extra_args', nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # 初始化消息总线（如果启用）
    if MESSAGE_BUS_ENABLED:
        try:
            MESSAGE_BUS = MessageBus(
                redis_url=os.environ.get('REDIS_URL', 'redis://redis:6379'),
                service_name="finrl-deepseek-trainer"
            )
            CORRELATION_ID = f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
            print(f"[MessageBus] Enabled, correlation_id={CORRELATION_ID}")
        except Exception as e:
            print(f"[MessageBus] Failed to initialize: {e}")
            MESSAGE_BUS = None

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # 从配置文件读取超参数
    training_config = get_training_config()
    epochs = training_config['epochs']
    print(f"Starting CPPO-DeepSeek training with {epochs} epochs...")
    print(f"Hyperparameters loaded from config file")

    hyperparams = {
        "epochs": epochs,
        "seed": args.seed,
        "exp_name": args.exp_name
    }

    # 发布训练开始事件
    if MESSAGE_BUS_ENABLED:
        publish_event(
            EventType.TRAINING_STARTED,
            hyperparams=hyperparams
        )

    training_start_time = time.time()
    model_path = None

    try:
        trained_cppo = cppo(lambda: env_train, actor_critic=MLPActorCritic,
                            seed=args.seed, logger_kwargs=logger_kwargs,
                            env_val=env_val)

        # Save the model
        model_path = TRAINED_MODEL_DIR + f"/agent_cppo_deepseek_{epochs}_epochs.pth"
        torch.save(trained_cppo.state_dict(), model_path)
        print("Training finished and saved in " + model_path)

        # 发布训练完成事件
        if MESSAGE_BUS_ENABLED:
            training_duration = time.time() - training_start_time
            publish_event(
                EventType.TRAINING_COMPLETED,
                model_path=model_path,
                metrics={
                    "epochs": epochs,
                    "duration_seconds": round(training_duration, 2),
                    "duration_minutes": round(training_duration / 60, 1)
                },
                hyperparams=hyperparams
            )

    except Exception as e:
        # 发布训练失败事件
        if MESSAGE_BUS_ENABLED:
            publish_event(
                EventType.TRAINING_FAILED,
                error=str(e),
                hyperparams=hyperparams
            )
        raise
    finally:
        # 关闭消息总线
        if MESSAGE_BUS is not None:
            MESSAGE_BUS.close()
