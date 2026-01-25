#!/usr/bin/env python
# coding: utf-8
"""
Configuration Loader for FinRL-DeepSeek

从挂载的配置文件读取配置，遵循统一配置管理设计：
- 非敏感配置: 从 config/settings.yaml 读取
- 敏感数据: 通过 pkg.database 封装层获取（内部使用 Doppler）
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# 配置文件路径优先级
CONFIG_PATHS = [
    '/app/config/settings.yaml',           # Docker 挂载路径
    Path(__file__).parent.parent.parent / 'config' / 'settings.yaml',  # 开发环境相对路径
]

_config_cache: Optional[Dict[str, Any]] = None


def _load_config() -> Dict[str, Any]:
    """加载配置文件"""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    for path in CONFIG_PATHS:
        path = Path(path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                _config_cache = yaml.safe_load(f) or {}
                return _config_cache

    # 配置文件不存在，使用空配置（依赖默认值）
    _config_cache = {}
    return _config_cache


def get_paths_config() -> Dict[str, str]:
    """获取路径配置"""
    config = _load_config()
    paths = config.get('paths', {})
    return {
        'models_dir': paths.get('models_dir', '/app/models'),
        'results_dir': paths.get('results_dir', '/app/results'),
        'data_dir': paths.get('data_dir', '/app/data'),
    }


def get_training_config() -> Dict[str, Any]:
    """获取训练配置"""
    config = _load_config()
    training = config.get('training', {})
    return {
        'data_source': training.get('data_source', 'clickhouse'),
        'lookback_days': training.get('lookback_days', 1825),
        'epochs': training.get('epochs', 100),
        'test_ratio': training.get('test_ratio', 0.2),
    }


def get_models_dir() -> str:
    """获取模型目录"""
    return get_paths_config()['models_dir']


def get_data_source() -> str:
    """获取数据来源"""
    return get_training_config()['data_source']


def get_lookback_days() -> int:
    """获取训练数据回溯天数"""
    return get_training_config()['lookback_days']


def get_epochs() -> int:
    """获取训练轮数"""
    return get_training_config()['epochs']
