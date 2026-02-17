#!/usr/bin/env python
# coding: utf-8
"""
Configuration Loader for FinRL-DeepSeek

统一从 EnvManager 获取配置，遵循统一配置管理设计：
- 非敏感配置: EnvManager → config/settings.yaml
- 敏感数据: EnvManager → Doppler SDK
"""

import os
import sys
from typing import Any, Dict

# 添加父项目路径，复用 utils.env_manager
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from utils.env_manager import EnvManager


def get_paths_config() -> Dict[str, str]:
    """获取路径配置 - 通过 EnvManager"""
    return EnvManager.paths_config()


def get_training_config() -> Dict[str, Any]:
    """获取训练配置 - 通过 EnvManager"""
    return EnvManager.training_config()


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
