#!/usr/bin/env python
# coding: utf-8
"""
llm_risk CVaR 消融实验 —— 回答"DeepSeek 的 LLM 风险信号到底有没有 OOS alpha"

背景：2026-05-27 go/no-go 跑在无 CVaR 机制的 auto_optimize 路径上（评估有效性
事故），其"中庸 +0.179"结论不评判完整的 FinRL-DeepSeek。本脚本在唯一算法核心
cppo_core 上跑严格对照：
  - 臂 A: use_llm_risk_cvar=True  （完整 LLM-risk-adjusted CVaR-PPO）
  - 臂 B: use_llm_risk_cvar=False （llm_risk_factor≡1，退化标准 PPO）
两臂同跨期 OOS split / 同超参 / 同 seed，唯一变量 = LLM 风险信号是否驱动 CVaR。

详见 docs/decisions/2026-05-28-finrl-deepseek-evaluation-validity-incident.md

运行（K8s GPU，单进程，不用 mpirun）：
  python3 ablation_llm_risk.py [--epochs 60] [--seed 0]
"""

import os
import argparse
import logging

import numpy as np
import pandas as pd

import cppo_core

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("ablation_llm_risk")


def load_data_split():
    """跨期 temporal split：早期 80% 训练 / 后期 20% OOS 验证（无 lookahead）。

    与 auto_optimize.load_data 同源逻辑。DATA_SOURCE=clickhouse 时走 ClickHouse，
    否则 HuggingFace。
    """
    data_source = os.environ.get('DATA_SOURCE', 'huggingface').lower()
    if data_source == 'clickhouse':
        from clickhouse_data_adapter import load_training_data
        start = os.environ.get('TRAIN_START_DATE', '2022-06-01')
        end = os.environ.get('TRAIN_END_DATE', '2025-07-01')
        train_df, val_df = load_training_data(start_date=start, end_date=end, test_ratio=0.2)
        logger.info(f"ClickHouse: train {len(train_df)} / val {len(val_df)} 行")
        return train_df, val_df

    from datasets import load_dataset
    dataset = load_dataset("benstaf/nasdaq_2013_2023",
                           data_files="train_data_deepseek_risk_2013_2018.csv")
    df = pd.DataFrame(dataset['train'])
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    unique_dates = sorted(df['date'].unique())
    split_date = unique_dates[int(len(unique_dates) * 0.8)]
    train_df = df[df['date'] < split_date].reset_index(drop=True)
    val_df = df[df['date'] >= split_date].reset_index(drop=True)
    logger.info(f"HuggingFace: train<{split_date} {len(train_df)} / val {len(val_df)} 行")
    return train_df, val_df


# Round 11 warm-start "历史最佳" 超参（两臂共用，隔离 llm_risk 为唯一变量）
SHARED_HP = dict(
    epochs=180,
    gamma=0.972709116752409,
    clip_ratio=0.11227346352610473,
    pi_lr=2.0131428489540617e-05,
    vf_lr=0.0005678133375978763,
    train_pi_iters=33,
    train_v_iters=7,
    lam=0.9899690665828681,
    target_kl=0.04719432475234017,
    hmax=194,
    hidden_sizes=(64, 64),
)


def run_arm(train_df, val_df, *, use_llm_risk_cvar, seed, epochs):
    label = "ON (LLM-risk CVaR)" if use_llm_risk_cvar else "OFF (vanilla PPO)"
    logger.info("=" * 60)
    logger.info(f"消融臂: use_llm_risk_cvar={use_llm_risk_cvar} — {label}")
    logger.info("=" * 60)

    hp = dict(SHARED_HP)
    hp['epochs'] = epochs

    train_vec, stock_dim = cppo_core.build_env(
        train_df, hmax=hp['hmax'], reward_scaling=6.0879465081638936e-05, vec=True)
    val_env, _ = cppo_core.build_env(
        val_df, hmax=hp['hmax'], reward_scaling=6.0879465081638936e-05, vec=False)

    _, metrics = cppo_core.cppo_train(
        lambda: train_vec,
        stock_dim,
        seed=seed,
        use_llm_risk_cvar=use_llm_risk_cvar,
        val_env=val_env,
        eval_every=5,
        early_stop_patience=6,
        **hp,
    )
    logger.info(f"[{label}] OOS metrics: {metrics}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    train_df, val_df = load_data_split()

    # 同 seed 同超参，仅切 llm_risk 开关
    on = run_arm(train_df, val_df, use_llm_risk_cvar=True, seed=args.seed, epochs=args.epochs)
    off = run_arm(train_df, val_df, use_llm_risk_cvar=False, seed=args.seed, epochs=args.epochs)

    delta = on['score'] - off['score']
    logger.info("=" * 60)
    logger.info("消融结果对照（OOS held-out，同 split/超参/seed）")
    logger.info("-" * 60)
    logger.info(f"{'metric':<16}{'ON(CVaR)':>14}{'OFF(vanilla)':>16}{'Δ(ON-OFF)':>14}")
    for k in ('score', 'sharpe', 'total_return', 'max_dd'):
        logger.info(f"{k:<16}{on[k]:>14.4f}{off[k]:>16.4f}{on[k]-off[k]:>14.4f}")
    logger.info("-" * 60)
    verdict = ("LLM-risk 信号有正向 OOS 贡献" if delta > 0.02 else
               "LLM-risk 信号无显著 OOS 贡献（噪声/退化）" if abs(delta) <= 0.02 else
               "LLM-risk 信号 OOS 为负（有害）")
    logger.info(f"判读: Δscore={delta:+.4f} → {verdict}")
    logger.info("=" * 60)

    # 写 ParamStore 供追溯（fail-fast：写失败直接 raise）
    try:
        from pkg.params import update as _ps_update
        from datetime import datetime as _dt
        ts = _dt.now().isoformat()
        src = "ablation_llm_risk"
        _ps_update('finrl_deepseek.ablation_llm_risk_on_score', float(on['score']), source=src)
        _ps_update('finrl_deepseek.ablation_llm_risk_off_score', float(off['score']), source=src)
        _ps_update('finrl_deepseek.ablation_llm_risk_delta', float(delta), source=src)
        _ps_update('finrl_deepseek.ablation_llm_risk_ts', ts, source=src)
        logger.info("✅ ParamStore 已写入 ablation_llm_risk_* (4 keys)")
    except ImportError:
        logger.warning("pkg.params 不可用，跳过 ParamStore 写入（本地 dry-run）")


if __name__ == "__main__":
    main()
