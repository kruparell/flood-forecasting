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
import subprocess
import sys
from pathlib import Path

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


from googlehydrology.utils.gfile_utils import get_gfile
gfile = get_gfile()

import os

class GFileHandler(logging.StreamHandler):
    """Logging handler that writes to a gfile handle for CNS compatibility."""
    def __init__(self, filename, mode='a'):
        if gfile:
            self._file = gfile.GFile(filename, mode)
            super().__init__(self._file)
        else:
            super().__init__(sys.stdout)

    def emit(self, record):
        super().emit(record)
        if hasattr(self._file, 'flush'):
            self._file.flush()


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

    log_file_str = str(log_file)
    if log_file_str.startswith('/cns/'):
        if gfile:
            parent_dir = os.path.dirname(log_file_str)
            if not gfile.Exists(parent_dir):
                try:
                    gfile.MakeDirs(parent_dir)
                except Exception:
                    pass
            file_handler = GFileHandler(log_file_str)
        else:
            file_handler = logging.NullHandler()
    else:
        os.makedirs(os.path.dirname(os.path.abspath(log_file_str)), exist_ok=True)
        file_handler = logging.FileHandler(filename=log_file_str)

    stdout_handler = logging.StreamHandler(sys.stdout)

    logging.basicConfig(
        handlers=[file_handler, stdout_handler],
        level=level,
        style='{',
        datefmt='%H:%M:%S',
        format='[{levelname}] {asctime}.{msecs:0<3.0f} ({filename}:{funcName}) -- {message}',
    )

    # Make sure we log uncaught exceptions and print to stderr for Borg visibility
    def exception_logging(type, value, tb):
        LOGGER.exception(f'Uncaught exception', exc_info=(type, value, tb))
        sys.__excepthook__(type, value, tb)

    sys.excepthook = exception_logging

    LOGGER.info(f'Logging to {log_file} initialized.')

    # Suppress DEBUG-level logging from these modules:
    logging.getLogger('filelock').setLevel(logging.INFO)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    logging.getLogger('fsspec').setLevel(logging.INFO)
    logging.getLogger('zarr').setLevel(logging.INFO)

    logging_redirect_tqdm().__enter__()


def get_git_hash() -> str | None:
    """Get git commit hash of the project if it is a git repository.

    Returns
    -------
    str | None
        Git commit hash if project is a git repository, else None.
    """
    # get git commit hash if folder is a git repository
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
    except OSError:
        return None  # likely, git is not installed.


def save_git_diff(run_dir: Path):
    """Try to store the git diff to a file.

    Parameters
    ----------
    run_dir : Path
        Directory of the current run.
    """
    str_run_dir = str(run_dir)
    base_dir = str(Path(__file__).absolute().parent)
    try:
        # diff should include staged and unstaged changes, hence we use "HEAD"
        out = subprocess.check_output(
            ['git', '-C', base_dir, 'diff', 'HEAD'], stderr=subprocess.DEVNULL
        )
    except Exception:
        return

    new_diff = out.strip().decode('utf-8')
    if not new_diff:
        return

    if str_run_dir.startswith('/cns/'):
        from googlehydrology.utils.gfile_utils import get_gfile
        gfile = get_gfile()
        try:
            existing_diffs = gfile.Glob(f"{str_run_dir}/googlehydrology*.diff")
            file_path = f"{str_run_dir}/googlehydrology-{len(existing_diffs)}.diff"
            with gfile.GFile(file_path, 'w') as diff_file:
                diff_file.write(new_diff)
        except Exception as e:
            LOGGER.warning(f"Could not save git diff to CNS: {e}")
        return

    existing_diffs = list(run_dir.glob('googlehydrology*.diff'))
    if len(existing_diffs) > 0:
        last_diff_path = (
            run_dir / f'googlehydrology-{len(existing_diffs) - 1}.diff'
        )
        with last_diff_path.open('r') as last_diff_file:
            last_diff = last_diff_file.read()
        if last_diff == new_diff:
            LOGGER.info(
                f'Git repository contains uncommitted changes that are stored in {last_diff_path}.'
            )
            return

    file_path = run_dir / f'googlehydrology-{len(existing_diffs)}.diff'
    LOGGER.warning(
        f'Git repository contains uncommitted changes. Writing diff to {file_path}.'
    )
    with file_path.open('w') as diff_file:
        diff_file.write(new_diff)
