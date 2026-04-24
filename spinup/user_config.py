import os
import os.path as osp

# Default neural network backend for each algo
# (Must be either 'tf1' or 'pytorch')
DEFAULT_BACKEND = {
    'vpg': 'pytorch',
    'trpo': 'tf1',
    'ppo': 'pytorch',
    'ddpg': 'pytorch',
    'td3': 'pytorch',
    'sac': 'pytorch'
}

# Where experiment outputs (training checkpoints) are saved by default:
# CHECKPOINTS_DIR 必须是 local PVC（频繁读写、不需要跨节点共享）
# MODELS_DIR 是网络挂载的云共享存储，只放最终模型，不放 checkpoint
_checkpoints_dir = os.environ.get('CHECKPOINTS_DIR', '/app/checkpoints')
DEFAULT_DATA_DIR = osp.join(_checkpoints_dir, 'finrl_deepseek')

# Whether to automatically insert a date and time stamp into the names of
# save directories:
FORCE_DATESTAMP = False

# Whether GridSearch provides automatically-generated default shorthands:
DEFAULT_SHORTHAND = True

# Tells the GridSearch how many seconds to pause for before launching 
# experiments.
WAIT_BEFORE_LAUNCH = 5