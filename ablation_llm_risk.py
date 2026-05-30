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


def _load_hf_csv(data_file):
    """从 HuggingFace benstaf/nasdaq_2013_2023 加载单个 CSV，确保 llm 列存在。"""
    from datasets import load_dataset
    ds = load_dataset("benstaf/nasdaq_2013_2023", data_files=data_file)
    df = pd.DataFrame(ds['train'])
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    # 防御：某些 trade 文件可能缺 llm 列（HF viewer 报过 schema 不一致）
    if 'llm_sentiment' not in df.columns or 'llm_risk' not in df.columns:
        raise RuntimeError(f"{data_file} 缺 llm_sentiment/llm_risk 列 — 须用 deepseek_risk 版文件")
    return df


def load_data_split():
    """加载训练 / OOS 验证数据。

    OOS_MODE（默认 true_oos）：
      - true_oos: 对齐原版论文——train=train_data_deepseek_risk_2013_2018.csv 全部，
        val=trade_data_deepseek_risk_2019_2023.csv 全部（独立后续期，真样本外）。
      - internal_split: 旧行为，2013-2018 内部 80/20 切（同分布，非真 OOS）。

    DATA_SOURCE=clickhouse 时走 ClickHouse（消融不用，保留兼容）。
    """
    data_source = os.environ.get('DATA_SOURCE', 'huggingface').lower()
    if data_source == 'clickhouse':
        from clickhouse_data_adapter import load_training_data
        start = os.environ.get('TRAIN_START_DATE', '2022-06-01')
        end = os.environ.get('TRAIN_END_DATE', '2025-07-01')
        train_df, val_df = load_training_data(start_date=start, end_date=end, test_ratio=0.2)
        logger.info(f"ClickHouse: train {len(train_df)} / val {len(val_df)} 行")
        return train_df, val_df

    oos_mode = os.environ.get('OOS_MODE', 'true_oos').lower()
    if oos_mode == 'true_oos':
        # 对齐原版：独立的后续期 trade 文件做 OOS（真样本外）
        train_df = _load_hf_csv("train_data_deepseek_risk_2013_2018.csv")
        val_df = _load_hf_csv("trade_data_deepseek_risk_2019_2023.csv")
        logger.info(f"true_oos: train(2013-2018) {len(train_df)} / "
                    f"val(2019-2023, 独立后续期) {len(val_df)} 行")
        return train_df, val_df

    # internal_split: 旧行为（2013-2018 内部切，同分布非真 OOS）
    df = _load_hf_csv("train_data_deepseek_risk_2013_2018.csv")
    unique_dates = sorted(df['date'].unique())
    split_date = unique_dates[int(len(unique_dates) * 0.8)]
    train_df = df[df['date'] < split_date].reset_index(drop=True)
    val_df = df[df['date'] >= split_date].reset_index(drop=True)
    logger.info(f"internal_split: train<{split_date} {len(train_df)} / val {len(val_df)} 行")
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
    # hmax 是 env 参数（非算法参数）：单独 pop 给 build_env，不进 cppo_train
    hmax = hp.pop('hmax')
    reward_scaling = 6.0879465081638936e-05

    # 训练用 DummyVecEnv（CVaR rollout 靠 next_o[0,...] 取值）；
    # 评估用 raw env（evaluate_oos 读 raw_env.state / raw_env.stock_dim）
    train_vec, stock_dim = cppo_core.build_env(
        train_df, hmax=hmax, reward_scaling=reward_scaling, vec=True)
    val_env, _ = cppo_core.build_env(
        val_df, hmax=hmax, reward_scaling=reward_scaling, vec=False)

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
    logger.info("=" * 64)
    logger.info("消融结果对照（OOS held-out，同 split/超参/seed）")
    logger.info("-" * 64)
    logger.info(f"{'metric':<20}{'ON(CVaR)':>14}{'OFF(vanilla)':>16}{'Δ(ON-OFF)':>14}")
    for k in ('score', 'sharpe', 'total_return', 'max_dd',
              'information_ratio', 'bench_total_return', 'excess_return'):
        logger.info(f"{k:<20}{on[k]:>14.4f}{off[k]:>16.4f}{on[k]-off[k]:>14.4f}")
    logger.info("-" * 64)

    # 分年 IR（特别看 2022 熊市——论文称 CPPO-DeepSeek 熊市更优）
    logger.info("分年 Information Ratio（剥离 beta 的 alpha；2022=熊市）:")
    years = sorted(set(on['ir_by_year']) | set(off['ir_by_year']))
    logger.info(f"{'year':<20}{'ON(CVaR)':>14}{'OFF(vanilla)':>16}{'Δ':>14}")
    for y in years:
        o = on['ir_by_year'].get(y, 0.0)
        f = off['ir_by_year'].get(y, 0.0)
        tag = "  ← 熊市" if y == '2022' else ""
        logger.info(f"{y:<20}{o:>14.4f}{f:>16.4f}{o - f:>14.4f}{tag}")
    logger.info("-" * 64)

    # 判读 1：llm_risk 是否有 OOS 贡献（复合 score）
    verdict = ("LLM-risk 信号有正向 OOS 贡献" if delta > 0.02 else
               "LLM-risk 信号无显著 OOS 贡献（噪声/退化）" if abs(delta) <= 0.02 else
               "LLM-risk 信号 OOS 为负（有害）")
    logger.info(f"判读1(llm_risk): Δscore={delta:+.4f} → {verdict}")

    # 判读 2：是否产出 alpha（IR>0 且跑赢基准），还是只是 beta
    on_ir, on_exc = on['information_ratio'], on['excess_return']
    if on_ir > 0.05 and on_exc > 0:
        alpha_verdict = f"产出正 alpha（IR={on_ir:.4f}>0 且跑赢大盘 {on_exc:+.2%}）→ 值得深究"
    elif on_ir <= 0.05 and on_exc <= 0:
        alpha_verdict = (f"是 beta 非 alpha（IR={on_ir:.4f}≈0 且未跑赢大盘 {on_exc:+.2%}）→ 不投产")
    else:
        alpha_verdict = (f"边缘（IR={on_ir:.4f}, 超额={on_exc:+.2%}）→ 看熊市段 IR_2022={on['ir_bear_2022']:+.4f}")
    logger.info(f"判读2(alpha vs beta): {alpha_verdict}")

    # 判读 3：熊市抗跌价值（论文唯一站得住的卖点）
    on_bear, off_bear = on['ir_bear_2022'], off['ir_bear_2022']
    bear_verdict = ("CPPO-DeepSeek 熊市抗跌有价值（2022 IR 显著高于 vanilla）→ 可作防御腿"
                    if on_bear - off_bear > 0.05 and on_bear > 0 else
                    "熊市段 llm_risk 无显著抗跌优势 → 论文卖点未复现")
    logger.info(f"判读3(熊市抗跌): ON 2022 IR={on_bear:+.4f} vs OFF {off_bear:+.4f} → {bear_verdict}")
    logger.info("=" * 64)

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
