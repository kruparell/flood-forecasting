"""Smoke test for model evaluation and dataset inference."""

import sys
from pathlib import Path
import torch
import xarray as xr

_SCRIPT_DIR = Path(__file__).resolve().parent
_TUTORIAL_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _TUTORIAL_DIR.parent

for _p in [_REPO_ROOT, _TUTORIAL_DIR, _SCRIPT_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from googlehydrology.datasetzoo.multimet import _convert_to_tensor
from googlehydrology.evaluation.tester import RegressionTester
from googlehydrology.utils.config import Config


def test_model_eval():
    run_dir_me = _TUTORIAL_DIR / 'model-runs' / 'generic-meanembedding-50basin_2107_080323'
    if (run_dir_me / 'config.yml').exists():
        cfg_me = Config(run_dir_me / 'config.yml')
        tester_me = RegressionTester(cfg=cfg_me, run_dir=run_dir_me, period='test', init_model=True)
        sample = tester_me.dataset[0]
        batch = tester_me.dataset.collate_fn([{k: _convert_to_tensor(k, v) for k, v in sample.items()}])
        tester_me.model.eval()
        with torch.no_grad():
            out = tester_me.model(batch)
        assert 'y_hat' in out
        assert out['y_hat'].shape[0] > 0


if __name__ == '__main__':
    test_model_eval()
    print("test_model_eval passed successfully!")
