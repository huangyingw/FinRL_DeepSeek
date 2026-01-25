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
import warnings
warnings.filterwarnings('ignore')

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

# Force GPU usage
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This training requires GPU.")
DEVICE = torch.device("cuda")
print(f"Using device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0)}")


def load_data():
    """根据配置文件加载训练数据"""
    training_config = get_training_config()
    data_source = training_config['data_source'].lower()

    if data_source == 'clickhouse':
        print("Loading training data from ClickHouse...")
        try:
            from clickhouse_data_adapter import load_training_data
            lookback_days = training_config['lookback_days']
            train, _ = load_training_data(lookback_days=lookback_days, test_ratio=0.0)
            print(f"Loaded {len(train)} rows from ClickHouse (lookback_days={lookback_days})")
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

    return train


# Load data
train = load_data()

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

e_train_gym = StockTradingEnv(df=train, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()


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
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
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
        self.rew_buf[self.ptr] = rew.item()
        self.val_buf[self.ptr] = val.item()
        self.valupdate_buf[self.ptr] = valupdate.item()
        self.logp_buf[self.ptr] = logp.item()
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        self.adv_buf = self.adv_buf - self.valupdate_buf
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
    epochs = epochs or int(BEST_PARAMS.get('epochs', 100))
    gamma = gamma or float(BEST_PARAMS.get('gamma', 0.995))
    clip_ratio = clip_ratio or float(BEST_PARAMS.get('clip_ratio', 0.7))
    pi_lr = pi_lr or float(BEST_PARAMS.get('pi_lr', 3e-5))
    vf_lr = vf_lr or float(BEST_PARAMS.get('vf_lr', 1e-4))
    train_pi_iters = train_pi_iters or int(BEST_PARAMS.get('train_pi_iters', 100))
    train_v_iters = train_v_iters or int(BEST_PARAMS.get('train_v_iters', 100))
    lam = lam or float(BEST_PARAMS.get('lam', 0.95))
    target_kl = target_kl or float(BEST_PARAMS.get('target_kl', 0.35))

    # 网络结构：从 best_params.json 读取（默认 512x512）
    hidden_size_1 = int(BEST_PARAMS.get('hidden_size_1', 512))
    hidden_size_2 = int(BEST_PARAMS.get('hidden_size_2', 512))
    ac_kwargs = ac_kwargs or dict(hidden_sizes=[hidden_size_1, hidden_size_2], activation=torch.nn.ReLU)

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

    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

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
            mpi_avg_grads(ac.pi)
            pi_optimizer.step()

        logger.store(StopIter=i)

        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)
            vf_optimizer.step()

        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    for epoch in range(epochs):
        trajectory_num = 0
        bad_trajectory_num = 0
        cvarlam = cvarlam + lam_lr * (beta - nu)
        lam_delta = 0
        nu_delta = 0
        update_num = 0

        for t in range(local_steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32, device=DEVICE))

            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            llm_risks = np.array(next_o[0, -stock_dimension:])

            risk_to_weight = {1: 0.99, 2: 0.995, 3: 1.0, 4: 1.005, 5: 1.01}
            llm_risks_weights = np.vectorize(risk_to_weight.get)(llm_risks)

            prices = np.array(next_o[0, 1:stock_dimension+1])
            shares = np.array(next_o[0, stock_dimension+1:stock_dimension*2+1])

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
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32, device=DEVICE))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0

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

        print("-" * 37)
        print("bad_trajectory_num:", bad_trajectory_num)
        print("update num:", update_num)
        print("nu:", nu)
        print("lam:", cvarlam)
        print("-" * 37, flush=True)
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

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # 从配置文件读取超参数
    training_config = get_training_config()
    epochs = training_config['epochs']
    print(f"Starting CPPO-DeepSeek training with {epochs} epochs...")
    print(f"Hyperparameters loaded from config file")

    trained_cppo = cppo(lambda: env_train, actor_critic=MLPActorCritic,
                        seed=args.seed, logger_kwargs=logger_kwargs)

    # Save the model
    model_path = TRAINED_MODEL_DIR + f"/agent_cppo_deepseek_{epochs}_epochs.pth"
    torch.save(trained_cppo.state_dict(), model_path)
    print("Training finished and saved in " + model_path)
