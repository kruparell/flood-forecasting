import os
import sys
import shutil
from pathlib import Path
import yaml
import time

repo_root = Path("/usr/local/google/home/kruparell/flood-forecasting")
tut_dir = repo_root / "tutorial"

sys.path.insert(0, str(repo_root))
from googlehydrology.utils.config import Config
from googlehydrology.training.train import start_training
from googlehydrology.evaluation.evaluate import start_evaluation

target_run_dir = tut_dir / "model-runs" / "arlstm-50basin-example_2107_080318"
cfg_file = target_run_dir / "config.yml"
with open(cfg_file, "r") as f:
    cfg_dict = yaml.safe_load(f)

cfg_dict["run_dir"] = str(target_run_dir)
cfg_dict["train_dir"] = str(target_run_dir / "train_data")
cfg_dict["img_log_dir"] = str(target_run_dir / "img_log")
cfg_dict["experiment_name"] = "arlstm-50basin-example"
cfg_dict["epochs"] = 10
cfg_dict["batch_size"] = 256
cfg_dict["max_updates_per_epoch"] = 5
cfg_dict["validate_every"] = 10
cfg_dict["num_workers"] = 0

with open(cfg_file, "w") as f:
    yaml.dump(cfg_dict, f)

cfg = Config(cfg_file)

t0 = time.time()
print(f"Starting 10-epoch ARLSTM training with num_workers=0 in {target_run_dir}...")
start_training(cfg)
t_train = time.time() - t0
print(f"Training completed in {t_train:.2f}s!")

t0 = time.time()
print("Starting evaluation for epoch 10...")
start_evaluation(cfg, run_dir=target_run_dir, epoch=10, period="test")
t_eval = time.time() - t0
print(f"Evaluation completed in {t_eval:.2f}s!")
