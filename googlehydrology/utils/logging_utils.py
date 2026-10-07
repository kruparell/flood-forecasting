# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from pathlib import Path
import subprocess
import sys

from tqdm.contrib.logging import logging_redirect_tqdm

LOGGER = logging.getLogger(__name__)


class WarningOnceFilter(logging.Filter):
    """Filters out non-unique warnings."""

    def __init__(self) -> None:
        super().__init__()
        self._seen = set()

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno != logging.WARNING:
            return True  # No effect for non-warning

        if (cmp := repr(record)) in self._seen:
            return False  # Filter out (already seen)
        self._seen.add(cmp)

        return True  # First seen so printed


def setup_logging(log_file: str, level: int, print_warnings_once: bool):
  """Initialize logging to `log_file` and stdout.

    Parameters
    ----------
    log_file : str
        Name of the file that will be logged to.
    level : int
        Py logging level to print from from.
    print_warnings_once : bool
        Whether to filter warnings that same line type and msg.
    """
  logging.captureWarnings(True)
  if print_warnings_once:
    logging.getLogger('py.warnings').addFilter(WarningOnceFilter())

  handlers = [logging.StreamHandler(sys.stdout)]
  if log_file and not str(log_file).startswith('/cns/'):
    try:
      import os

      parent = os.path.dirname(str(log_file))
      if parent:
        os.makedirs(parent, exist_ok=True)
      handlers.append(logging.FileHandler(filename=log_file))
    except Exception:
      pass

  logging.basicConfig(
      handlers=handlers,
      level=level,
      style='{',
      datefmt='%H:%M:%S',
      format=(
          '[{levelname}] {asctime}.{msecs:0<3.0f} ({filename}:{funcName}) --'
          ' {message}'
      ),
      force=True,
  )

  # Make sure we log uncaught exceptions
  def exception_logging(type, value, tb):
    LOGGER.exception(f'Uncaught exception', exc_info=(type, value, tb))

  sys.excepthook = exception_logging

  LOGGER.info(f'Logging to {log_file} initialized.')

  # Suppress DEBUG-level logging from these modules:
  logging.getLogger('filelock').setLevel(logging.INFO)
  logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
  logging.getLogger('fsspec').setLevel(logging.INFO)
  logging.getLogger('zarr').setLevel(logging.INFO)

  logging_redirect_tqdm().__enter__()


def get_git_hash() -> str | None:
  """Get git commit hash of the project if it is a git repository."""
  current_dir = str(Path(__file__).absolute().parent)
  try:
    if (
        subprocess.call(
            ['git', '-C', current_dir, 'branch'],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )
        == 0
    ):
      return (
                subprocess.check_output(
                    ['git', '-C', current_dir, 'describe', '--always']
                )
                .strip()
                .decode('ascii')
            )
  except Exception:
    return None
  return None


def save_git_diff(run_dir: Path):
  """Try to store the git diff to a file."""
  base_dir = str(Path(__file__).absolute().parent)
  try:
    out = subprocess.check_output(
        ['git', '-C', base_dir, 'diff', 'HEAD'], stderr=subprocess.DEVNULL
    )
    new_diff = out.strip().decode('utf-8')
    if new_diff and not str(run_dir).startswith('/cns/'):
      file_path = run_dir / 'googlehydrology.diff'
      with open(str(file_path), 'w') as diff_file:
        diff_file.write(new_diff)
  except Exception:
    pass
