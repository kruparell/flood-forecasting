import os
import sys
import time
from pathlib import Path
import torch
import yaml

repo_root = Path('/usr/local/google/home/kruparell/flood-forecasting')
sys.path.insert(0, str(repo_root))

from googlehydrology.utils.config import Config
from googlehydrology.run import finetune, eval_run

base_run_dir = repo_root / 'tutorial' / 'model-runs' / 'generic-meanembedding-50basin_2107_080323'
config_path = repo_root / 'tutorial' / 'configs' / 'finetune-generic-meanembedding-50basin.yml'

finetune_config = {
    'base_run_dir': str(base_run_dir),
    'run_dir': str(base_run_dir),
    'experiment_name': 'finetune-50basin',
    'finetune_modules': ['static_embedding_fc', 'head'],
    'train_basin_file': str(repo_root / 'tutorial' / 'basin-lists' / '50-basin-train.txt'),
    'validation_basin_file': str(repo_root / 'tutorial' / 'basin-lists' / '50-basin-train.txt'),
    'test_basin_file': str(repo_root / 'tutorial' / 'basin-lists' / '50-basin-train.txt'),
    'targets_data_dir': '/usr/local/google/home/kruparell/Caravans/Caravan-nc',
    'dynamics_data_dir': '/usr/local/google/home/kruparell/Caravans_MultiMet',
    'epochs': 2,
    'batch_size': 256,
    'initial_learning_rate': 0.005,
    'learning_rate_strategy': 'StepLR',
    'learning_rate_drop_factor': 0.9,
    'learning_rate_epochs_drop': 5,
    'max_updates_per_epoch': 50,
    'metrics': ['NSE', 'KGE'],
    'validate_every': 1,
    'validate_n_random_basins': -1,
}

with open(config_path, 'w') as f:
    yaml.dump(finetune_config, f, default_flow_style=False)

print(f"Created finetune config at {config_path}")
