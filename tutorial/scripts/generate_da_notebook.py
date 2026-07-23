#!/usr/bin/env python3
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

"""Delegates to generate_and_render_da_notebook.py for authentic data assimilation execution."""

import os
import sys
from pathlib import Path

repo_root = Path('/usr/local/google/home/kruparell/flood-forecasting')
scripts_dir = repo_root / 'tutorial' / 'scripts'
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

import generate_and_render_da_notebook

if __name__ == '__main__':
    print("Executing authentic Data Assimilation notebook generation...")
    # Executing the canonical data assimilation generator
    os.system(f"{sys.executable} {scripts_dir / 'generate_and_render_da_notebook.py'}")
