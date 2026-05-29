#!/usr/bin/env python
# coding: utf-8
"""
cppo_core —— FinRL-DeepSeek CVaR-PPO 算法核心（唯一真相源）

背景：此前生产训练（train_cppo_llm_risk_standalone.py）与 HPO/OOS 评估
（auto_optimize.py）各有一份独立的 PPO 实现，且只有 standalone 那份带
LLM-risk-adjusted CVaR 机制。go/no-go 跑在无 CVaR 的 auto_optimize 上，
评判了一个退化策略 → 评估有效性事故。
详见 docs/decisions/2026-05-28-finrl-deepseek-evaluation-validity-incident.md

本模块把 standalone 的 CVaR-PPO 算法体**逐字搬运**为唯一实现，两个入口
（生产 / HPO）共用，消灭"两份会漂移的实现"这一根因。被测的 CVaR 机制
代码零重写。

相对 standalone 的唯一改动（均不触碰 CVaR 数学）：
1. 超参由调用方显式传入（不再读模块级全局 BEST_PARAMS）
2. stock_dimension 由调用方传入（原为模块级全局）
3. 新增 use_llm_risk_cvar 开关：False 时 llm_risk_factor≡1，CVaR 退化为
   标准 PPO —— 这是 llm_risk 消融的 on/off 控制
4. 新增可选 val_env 跨期 OOS 评估 + 早停 + 返回 metrics（供 HPO 路径用）

运行：单进程（不经 mpirun）。spinup 的 mpi_* 调用在 num_procs()==1 时
退化为恒等操作，无需改动算法。
"""

import os
import time
import numpy as np
import pandas as pd
import scipy.signal

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from gymnasium.spaces import Box, Discrete

import spinup.algos.pytorch.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_avg, proc_id, mpi_statistics_scalar, num_procs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INDICATORS = [
    'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
    'close_30_sma', 'close_60_sma',
]


# ============================================================
# 环境构造（共享脚手架，搬自 auto_optimize.create_env）
# ============================================================
def build_env(df, *, hmax=100, initial_amount=1000000, reward_scaling=1e-4,
              reward_type='pnl', dsr_eta=0.01, vec=False):
    """构造 StockTradingEnv。

    vec=True 返回 DummyVecEnv 包装（obs 形状 [1,dim]，训练用，CVaR rollout
    代码靠 next_o[0,...] 取值）；vec=False 返回 raw env（obs [dim]，OOS 评估用）。
    """
    from env_stocktrading_llm_risk import StockTradingEnv

    df = df.copy()
    unique_dates = df['date'].unique()
    date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
    df['new_idx'] = df['date'].map(date_to_idx)
    df = df.set_index('new_idx')

    df['llm_sentiment'] = df['llm_sentiment'].fillna(0)
    df['llm_risk'] = df['llm_risk'].fillna(3)

    stock_dimension = len(df.tic.unique())
    state_space = 1 + 2 * stock_dimension + (2 + len(INDICATORS)) * stock_dimension

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
        "reward_scaling": reward_scaling,
        "reward_type": reward_type,
        "dsr_eta": dsr_eta,
    }
    env = StockTradingEnv(df=df, **env_kwargs)
    if vec:
        e, _ = env.get_sb_env()
        return e, stock_dimension
    return env, stock_dimension


# ============================================================
# Neural Network Definitions（逐字搬自 standalone）
# ============================================================
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


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

    def act_deterministic(self, obs):
        """确定性动作（分布均值）——用于 OOS 评估，消除采样噪声。"""
        with torch.no_grad():
            return self.pi._distribution(obs).mean.cpu().numpy()


# ============================================================
# CPPO Buffer（逐字搬自 standalone）
# ============================================================
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


# ============================================================
# OOS 评估（搬自 auto_optimize 的 val 评估逻辑，中立脚手架）
# ============================================================
def evaluate_oos(ac, raw_env, initial_amount=1000000):
    """在 held-out raw env 上确定性评估，返回 OOS 指标。

    与训练用的 DummyVecEnv 不同，这里用未包装的 StockTradingEnv，obs 为 1D。
    复合 score 与 auto_optimize 一致：0.3*sharpe + 0.3*total_return - 0.4*max_dd
    """
    obs, _ = raw_env.reset()
    obs = np.asarray(obs, dtype=np.float32)
    done = False
    portfolio_values = [initial_amount]

    while not done:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
        action = ac.act_deterministic(obs_tensor)
        obs_raw, reward, terminated, truncated, _ = raw_env.step(action)
        obs = np.asarray(obs_raw, dtype=np.float32)
        done = terminated or truncated
        s = raw_env.state
        sd = raw_env.stock_dim
        pv = s[0] + sum(s[1 + i] * s[1 + sd + i] for i in range(sd))
        portfolio_values.append(pv)

    final_value = portfolio_values[-1]
    total_return = (final_value - initial_amount) / initial_amount

    pv_series = pd.Series(portfolio_values)
    daily_returns = pv_series.pct_change().dropna()
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * (252 ** 0.5)
    else:
        sharpe = 0.0
    cummax = pv_series.cummax()
    drawdown = ((pv_series - cummax) / cummax).min()
    max_dd = abs(drawdown) if not pd.isna(drawdown) else 0.0
    score = 0.3 * sharpe + 0.3 * total_return - 0.4 * max_dd

    return {
        'score': float(score),
        'sharpe': float(sharpe),
        'total_return': float(total_return),
        'max_dd': float(max_dd),
        'final_value': float(final_value),
    }


# ============================================================
# CVaR-PPO 训练核心（算法体逐字搬自 standalone cppo()）
#
# 改动仅限：超参显式入参 / stock_dimension 入参 / use_llm_risk_cvar 开关 /
# 可选 val_env 早停评估 / 返回 (ac, metrics)。CVaR 数学零改动。
# ============================================================
def cppo_train(env_fn,
               stock_dimension,
               *,
               actor_critic=MLPActorCritic,
               ac_kwargs=None,
               seed=42,
               steps_per_epoch=20000,
               epochs=100,
               gamma=0.995,
               clip_ratio=0.7,
               pi_lr=3e-5,
               vf_lr=1e-4,
               train_pi_iters=100,
               train_v_iters=100,
               lam=0.95,
               max_ep_len=3000,
               target_kl=0.35,
               hidden_sizes=(512, 512),
               activation=nn.ReLU,
               logger_kwargs=None,
               save_freq=10,
               # CVaR 超参（逐字搬自 standalone cppo 签名）
               alpha=0.85,
               beta=3000.0,
               nu_lr=5e-4,
               lam_lr=5e-4,
               nu_start=0.1,
               lam_start=0.01,
               nu_delay=0.75,
               lam_low_bound=0.001,
               delay=1.0,
               cvar_clip_ratio=0.05,
               # 合并新增（不触碰 CVaR 数学）
               use_llm_risk_cvar=True,
               val_env=None,
               eval_every=5,
               early_stop_patience=10,
               initial_amount=1000000):
    """训练 CVaR-PPO，返回 (ac, metrics)。

    use_llm_risk_cvar=False 时 llm_risk_factor 恒为 1，CVaR 风险加权退化为
    标准 PPO 优势——这是 llm_risk 消融的 off 臂。

    val_env 非空时：每 eval_every epoch 在 val_env 上做 OOS 评估，按复合
    score 早停并保留最优权重；metrics 含最优 OOS 指标。
    """
    ac_kwargs = ac_kwargs or dict(hidden_sizes=list(hidden_sizes), activation=activation)
    logger_kwargs = logger_kwargs or dict()

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
    ac = ac.to(DEVICE)
    sync_params(ac)

    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = CPPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    nu = nu_start
    cvarlam = lam_start

    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        return loss_pi, pi_info

    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret) ** 2).mean()

    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    logger.setup_pytorch_saver(ac)

    def update():
        # Diagnostic: do-nothing / value-collapse 排查（不改训练数学，仅记录）
        # 详见 docs/decisions/2026-05-18-finrl-deepseek-rl-instrumentation.md
        _act_raw = buf.act_buf.copy()
        _ret_var = float(buf.ret_buf.var())
        _residual_var = float((buf.ret_buf - buf.val_buf).var())
        _ev = 1.0 - _residual_var / (_ret_var + 1e-8)
        _act_mean_abs = float(np.abs(_act_raw).mean())
        _act_zero_frac = float((np.abs(_act_raw) < 0.01).mean())
        logger.store(ExplainedVar=_ev, ActMeanAbs=_act_mean_abs, ActZeroFrac=_act_zero_frac)

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

    # OOS 早停状态
    best_val_score = -float('inf')
    best_state = None
    no_improve = 0
    best_metrics = None

    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    risk_to_weight = {1: 0.99, 2: 0.995, 3: 1.0, 4: 1.005, 5: 1.01}

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

            if use_llm_risk_cvar:
                # llm_risks 已被 env._normalize_state 归一化除以 5，反归一化回
                # {1,2,3,4,5} 并 clamp 到有效范围
                llm_risks_normed = np.array(next_o[0, -stock_dimension:])
                llm_risks = np.clip(np.round(llm_risks_normed * 5).astype(int), 1, 5)
                llm_risks_weights = np.vectorize(lambda x: risk_to_weight.get(x, 1.0))(llm_risks)

                prices = np.array(next_o[0, 1:stock_dimension + 1])
                shares = np.array(next_o[0, stock_dimension + 1:stock_dimension * 2 + 1])

                stock_values = prices * shares
                total_value = np.sum(stock_values)
                if total_value == 0:
                    llm_risk_factor = 1
                else:
                    stock_weights = stock_values / total_value
                    llm_risk_factor = np.dot(stock_weights, llm_risks_weights)
            else:
                # 消融 off 臂：关闭 LLM 风险加权，CVaR 退化为标准 PPO 优势
                llm_risk_factor = 1

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

        update()

        # 跨期 OOS 评估 + 早停（仅 HPO 路径传入 val_env 时启用）
        if val_env is not None and (epoch % eval_every == 0 or epoch == epochs - 1):
            m = evaluate_oos(ac, val_env, initial_amount=initial_amount)
            logger.store(ValScore=m['score'], ValSharpe=m['sharpe'])
            if m['score'] > best_val_score:
                best_val_score = m['score']
                best_metrics = m
                best_state = {k: vv.detach().cpu().clone() for k, vv in ac.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= early_stop_patience:
                logger.log(f'OOS early stop at epoch {epoch} (best score {best_val_score:.4f})')
                break

        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ExplainedVar', average_only=True)
        logger.log_tabular('ActMeanAbs', average_only=True)
        logger.log_tabular('ActZeroFrac', average_only=True)
        if val_env is not None:
            logger.log_tabular('ValScore', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()

    # OOS 早停恢复最优权重
    if best_state is not None:
        ac.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    metrics = {
        'use_llm_risk_cvar': use_llm_risk_cvar,
        'epochs_trained': epoch + 1,
    }
    if best_metrics is not None:
        metrics.update(best_metrics)
    elif val_env is not None:
        metrics.update(evaluate_oos(ac, val_env, initial_amount=initial_amount))

    return ac, metrics
