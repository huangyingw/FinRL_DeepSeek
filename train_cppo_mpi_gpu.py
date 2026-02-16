#!/usr/bin/env python
# coding: utf-8
"""
CPPO-DeepSeek Training with MPI + GPU support
Fixes MPI environment sharing issues and forces GPU usage.

Run with:
    OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    mpirun -np 4 python3 train_cppo_mpi_gpu.py --epochs 100 --finetune

For fine-tuning from pretrained model:
    ... python3 train_cppo_mpi_gpu.py --epochs 20 --finetune --pretrained /path/to/model.pth
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import argparse
import numpy as np
import scipy.signal
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.normal import Normal
from gymnasium.spaces import Box, Discrete

# Force GPU usage
if not torch.cuda.is_available():
    print("WARNING: CUDA not available, falling back to CPU")
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# MPI imports - do this early
import spinup.algos.pytorch.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

# Stateless: 模型保存到 Docker named volume - 从配置文件读取
from config_loader import get_models_dir
SHARED_MODEL_DIR = os.path.join(get_models_dir(), 'finrl_deepseek')
os.makedirs(SHARED_MODEL_DIR, exist_ok=True)

# Indicators
INDICATORS = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma']


def load_data_and_create_env():
    """Load data and create environment. Called AFTER MPI init, only on rank 0 for data loading."""
    from datasets import load_dataset
    import pandas as pd
    from env_stocktrading_llm_risk import StockTradingEnv

    # Only rank 0 prints
    if proc_id() == 0:
        print("Loading training data from Hugging Face...")

    dataset = load_dataset("benstaf/nasdaq_2013_2023", data_files="train_data_deepseek_risk_2013_2018.csv")
    train = pd.DataFrame(dataset['train'])
    train = train.drop('Unnamed: 0', axis=1)

    unique_dates = train['date'].unique()
    date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
    train['new_idx'] = train['date'].map(date_to_idx)
    train = train.set_index('new_idx')

    train['llm_sentiment'].fillna(0, inplace=True)
    train['llm_risk'].fillna(3, inplace=True)

    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2*stock_dimension + (2+len(INDICATORS))*stock_dimension

    if proc_id() == 0:
        print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }

    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()

    return env_train, stock_dimension


# Neural Network definitions (same as before but on GPU)
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class MLPGaussianActor(nn.Module):
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

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()
        obs_dim = observation_space.shape[0]

        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)

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
        self.rew_buf[self.ptr] = rew.item() if hasattr(rew, 'item') else rew
        self.val_buf[self.ptr] = val.item() if hasattr(val, 'item') else val
        self.valupdate_buf[self.ptr] = valupdate.item() if hasattr(valupdate, 'item') else valupdate
        self.logp_buf[self.ptr] = logp.item() if hasattr(logp, 'item') else logp
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
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32, device=DEVICE) for k, v in data.items()}


def cppo(env_fn, stock_dimension, pretrained_path=None, finetune=False,
         ac_kwargs=dict(hidden_sizes=[512, 512], activation=torch.nn.ReLU),
         seed=42, steps_per_epoch=20000, epochs=100, gamma=0.995, clip_ratio=0.7,
         pi_lr=3e-5, vf_lr=1e-4, train_pi_iters=100, train_v_iters=100, lam=0.95,
         max_ep_len=3000, target_kl=0.35, logger_kwargs=dict(), save_freq=10,
         alpha=0.85, beta=3000.0, nu_lr=5e-4, lam_lr=5e-4, nu_start=0.1,
         lam_start=0.01, nu_delay=0.75, delay=1.0, cvar_clip_ratio=0.05):

    # MPI setup FIRST
    setup_pytorch_for_mpi()

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create environment (each process gets its own)
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic on GPU
    ac = MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs).to(DEVICE)

    # Load pretrained weights if fine-tuning
    if finetune and pretrained_path and os.path.exists(pretrained_path):
        if proc_id() == 0:
            print(f"Loading pretrained model from {pretrained_path}")
        pretrained_state = torch.load(pretrained_path, map_location=DEVICE)
        try:
            ac.load_state_dict(pretrained_state)
            if proc_id() == 0:
                print("Successfully loaded pretrained weights!")
            # Use lower learning rates for fine-tuning
            pi_lr = pi_lr / 3
            vf_lr = vf_lr / 3
        except RuntimeError as e:
            if proc_id() == 0:
                print(f"Warning: Could not load pretrained weights: {e}")

    # Sync params across MPI processes
    sync_params(ac)

    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    if proc_id() == 0:
        logger.log(f'\nNumber of parameters: pi: {var_counts[0]}, v: {var_counts[1]}\n')
        logger.log(f'Using device: {DEVICE}\n')

    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = CPPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    nu = nu_start
    cvarlam = lam_start

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
        return loss_pi, dict(kl=approx_kl, ent=ent, cf=clipfrac)

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
                if proc_id() == 0:
                    logger.log(f'Early stopping at step {i} due to reaching max kl.')
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
        logger.store(LossPi=pi_l_old, LossV=v_l_old, KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old), DeltaLossV=(loss_v.item() - v_l_old))

    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    for epoch in range(epochs):
        trajectory_num = 0
        bad_trajectory_num = 0
        cvarlam = cvarlam + lam_lr * (beta - nu)
        nu_delta = 0
        update_num = 0

        for t in range(local_steps_per_epoch):
            # Move observation to GPU
            o_tensor = torch.as_tensor(o, dtype=torch.float32, device=DEVICE)
            a, v, logp = ac.step(o_tensor)

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
                if epoch_ended and not terminal and proc_id() == 0:
                    print(f'Warning: trajectory cut off at {ep_len} steps.', flush=True)
                if timeout or epoch_ended:
                    o_tensor = torch.as_tensor(o, dtype=torch.float32, device=DEVICE)
                    _, v, _ = ac.step(o_tensor)
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0

        if trajectory_num > 0:
            nu_delta = nu_delta / trajectory_num
        nu = nu_delta * nu_delay

        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, None)

        update()

        if proc_id() == 0:
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

            print(f"--- Epoch {epoch} | bad_trajectories: {bad_trajectory_num} | nu: {nu:.4f} ---")

    return ac


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPPO-DeepSeek Training with MPI + GPU")
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--hid', type=int, default=512, help='Hidden layer size')
    parser.add_argument('--l', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed')
    parser.add_argument('--finetune', action='store_true', help='Fine-tune from pretrained model')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained model')
    parser.add_argument('--exp_name', type=str, default='cppo_deepseek_mpi_gpu', help='Experiment name')
    parser.add_argument('-f', '--file', type=str, help='Kernel connection file')

    args = parser.parse_args()

    # Default pretrained model path
    if args.finetune and args.pretrained is None:
        args.pretrained = os.path.join(SHARED_MODEL_DIR, 'agent_cppo_deepseek_100_epochs.pth')

    # Create environment function (called after MPI init)
    def make_env():
        env, stock_dim = load_data_and_create_env()
        return env

    # Get stock dimension for risk calculation
    env_temp, stock_dimension = load_data_and_create_env()
    del env_temp

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    if proc_id() == 0:
        print(f"Starting CPPO-DeepSeek training with {args.epochs} epochs...")
        print(f"Hidden layers: {args.hid} x {args.l}")
        print(f"Fine-tuning: {args.finetune}")
        if args.finetune:
            print(f"Pretrained model: {args.pretrained}")

    trained_cppo = cppo(
        make_env, stock_dimension,
        pretrained_path=args.pretrained if args.finetune else None,
        finetune=args.finetune,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        seed=args.seed, epochs=args.epochs, logger_kwargs=logger_kwargs
    )

    # Save model to shared volume (only rank 0)
    if proc_id() == 0:
        model_name = f"agent_cppo_{'finetuned' if args.finetune else 'trained'}_{args.epochs}_epochs.pth"
        model_path = os.path.join(SHARED_MODEL_DIR, model_name)
        torch.save(trained_cppo.state_dict(), model_path)
        print(f"\nTraining finished! Model saved to: {model_path}")
