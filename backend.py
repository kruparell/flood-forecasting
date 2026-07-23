# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hydrology model evaluation, metrics calculation, and visualization backend."""

import importlib.util
from pathlib import Path
import sys

_CURRENT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = next(
    (p for p in [_CURRENT_DIR, _CURRENT_DIR.parent] if (p / 'googlehydrology').is_dir()),
    _CURRENT_DIR
)
_TUTORIAL_DIR = _REPO_ROOT / 'tutorial' if (_REPO_ROOT / 'tutorial').is_dir() else _CURRENT_DIR
_SCRIPTS_DIR = _TUTORIAL_DIR / 'scripts'
_NOTEBOOKS_DIR = _TUTORIAL_DIR / 'notebooks'

for _p in [str(_REPO_ROOT), str(_TUTORIAL_DIR), str(_SCRIPTS_DIR), str(_NOTEBOOKS_DIR), str(_TUTORIAL_DIR / 'src')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import and expose all attributes directly from scripts/backend.py
_script_backend = _SCRIPTS_DIR / 'backend.py'
if _script_backend.exists():
    _spec = importlib.util.spec_from_file_location("tutorial_scripts_backend", str(_script_backend))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    for _k, _v in _mod.__dict__.items():
        if not _k.startswith('__'):
            globals()[_k] = _v
