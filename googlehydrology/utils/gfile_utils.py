# Copyright 2026 Google LLC
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

"""Helper utilities for robust gfile and CNS distributed filesystem operations."""

import collections.abc
import functools
import os
from pathlib import Path
import shutil
from typing import Any, IO, Iterator, Union

import torch


def get_gfile():
  """Safely acquires gfile in Google3, Borglet, and standalone environments."""
  try:
    from google3.pyglib import gfile

    return gfile
  except ImportError:
    try:
      from pyglib import gfile

      return gfile
    except ImportError:
      try:
        import tensorflow.io.gfile as gfile

        return gfile
      except ImportError:
        return None


_STRICT_ZARR_ONLY = os.environ.get("STRICT_ZARR_ONLY", "0").lower() in (
    "1",
    "true",
    "yes",
)


def is_strict_zarr_only() -> bool:
  """Returns True if strict Zarr-only policy is enabled."""
  return _STRICT_ZARR_ONLY


def set_strict_zarr_only(enabled: bool) -> None:
  """Globally enables or disables strict Zarr-only enforcement."""
  global _STRICT_ZARR_ONLY
  _STRICT_ZARR_ONLY = enabled


def is_cns_path(path: Union[str, Path]) -> bool:
  """Checks if a given path is a CNS distributed path."""
  return str(path).startswith("/cns/")


def gfile_exists(path: Union[str, Path]) -> bool:
  """Checks file/directory existence on CNS or local POSIX filesystem."""
  p_str = str(path)
  gf = get_gfile()
  if is_cns_path(p_str):
    if gf is None:
      raise RuntimeError(
          f"Cannot access CNS path '{p_str}': google3.pyglib.gfile is"
          " unavailable."
      )
    return gf.Exists(p_str)
  if gf is not None:
    return gf.Exists(p_str)
  return os.path.exists(p_str)


def gfile_open(
    path: Union[str, Path], mode: str = "r", encoding: str | None = None
) -> IO[Any]:
  """Opens a file on CNS or local filesystem."""
  p_str = str(path)
  gf = get_gfile()
  if is_cns_path(p_str):
    if gf is None:
      raise RuntimeError(
          f"Cannot open CNS path '{p_str}': google3.pyglib.gfile is"
          " unavailable."
      )
    return gf.GFile(p_str, mode)
  if gf is not None:
    return gf.GFile(p_str, mode)
  return open(p_str, mode, encoding=encoding)


def gfile_listdir(path: Union[str, Path]) -> list[str]:
  """Lists files/directories inside a path."""
  p_str = str(path)
  gf = get_gfile()
  if is_cns_path(p_str):
    if gf is None:
      raise RuntimeError(
          f"Cannot list CNS directory '{p_str}': google3.pyglib.gfile is"
          " unavailable."
      )
    raw = gf.ListDir(p_str)
    return [e.decode("utf-8") if isinstance(e, bytes) else str(e) for e in raw]
  if gf is not None:
    raw = gf.ListDir(p_str)
    return [e.decode("utf-8") if isinstance(e, bytes) else str(e) for e in raw]
  return [
      e.decode("utf-8") if isinstance(e, bytes) else str(e)
      for e in os.listdir(p_str)
  ]


def gfile_isdir(path: Union[str, Path]) -> bool:
  """Checks if a path is a directory on CNS or local filesystem."""
  p_str = str(path)
  gf = get_gfile()
  if is_cns_path(p_str):
    if gf is None:
      raise RuntimeError(
          f"Cannot check CNS directory '{p_str}': google3.pyglib.gfile is"
          " unavailable."
      )
    return gf.IsDirectory(p_str)
  if gf is not None:
    try:
      return gf.IsDirectory(p_str)
    except Exception:
      pass
  return os.path.isdir(p_str)


def gfile_glob(pattern: str) -> list[str]:
  """Glob match on CNS or local filesystem."""
  gf = get_gfile()
  if is_cns_path(pattern):
    if gf is None:
      raise RuntimeError(
          f"Cannot glob CNS pattern '{pattern}': google3.pyglib.gfile is"
          " unavailable."
      )
    return gf.Glob(pattern)
  if gf is not None:
    return gf.Glob(pattern)
  import glob

  return glob.glob(pattern)


def gfile_makedirs(path: Union[str, Path], exist_ok: bool = True) -> None:
  """Recursively creates directories on CNS or local filesystem."""
  p_str = str(path)
  gf = get_gfile()
  if is_cns_path(p_str):
    if gf is None:
      raise RuntimeError(
          f"Cannot make CNS directory '{p_str}': google3.pyglib.gfile is"
          " unavailable."
      )
    gf.MakeDirs(p_str)
    return
  if gf is not None:
    gf.MakeDirs(p_str)
    return
  os.makedirs(p_str, exist_ok=exist_ok)


def gfile_rmtree(path: Union[str, Path]) -> None:
  """Recursively removes a directory on CNS or local filesystem."""
  p_str = str(path)
  if not gfile_exists(p_str):
    return
  gf = get_gfile()
  if is_cns_path(p_str):
    if gf is None:
      raise RuntimeError(
          f"Cannot remove CNS directory '{p_str}': google3.pyglib.gfile is"
          " unavailable."
      )
    try:
      gf.DeleteRecursively(p_str)
    except Exception:
      pass
    return
  if gf is not None:
    try:
      gf.DeleteRecursively(p_str)
    except Exception:
      pass
    return
  shutil.rmtree(p_str, ignore_errors=True)


@functools.lru_cache(maxsize=1024)
def find_cns_subdir(base_dir: str, target_name: str) -> str | None:
  """Performs exact case-insensitive directory resolution on CNS."""
  gf = get_gfile()
  if gf and is_cns_path(base_dir):
    try:
      subdirs = gfile_listdir(base_dir)
      target_lower = target_name.lower()
      for d in subdirs:
        name = d.rstrip("/")
        if name.lower() == target_lower:
          full = f"{base_dir}/{name}"
          if gfile_isdir(full):
            return full
    except Exception:
      return None
  return None


class GFileZarrStore(collections.abc.MutableMapping):
  """A MutableMapping key-value store for Zarr datasets backed by google3.pyglib.gfile."""

  def __init__(self, root_path: Union[str, Path], mode: str = "r"):
    self.root_path = str(root_path).rstrip("/")
    self.mode = mode
    self.gf = get_gfile()

  def _full_path(self, key: str) -> str:
    return f"{self.root_path}/{key.lstrip('/')}"

  def __getitem__(self, key: str) -> bytes:
    p = self._full_path(key)
    try:
      with gfile_open(p, "rb") as f:
        return f.read()
    except Exception as e:
      raise KeyError(f"Key '{key}' not found in GFileZarrStore at {p}") from e

  def __setitem__(self, key: str, value: bytes) -> None:
    p = self._full_path(key)
    parent = os.path.dirname(p)
    gfile_makedirs(parent, exist_ok=True)
    with gfile_open(p, "wb") as f:
      f.write(value)

  def __delitem__(self, key: str) -> None:
    p = self._full_path(key)
    gf = get_gfile()
    if gf:
      gf.Remove(p)
    else:
      os.remove(p)

  def __iter__(self) -> Iterator[str]:
    gf = get_gfile()
    if is_cns_path(self.root_path) and gf:
      queue = [self.root_path]
      while queue:
        curr = queue.pop(0)
        try:
          for entry in gf.ListDir(curr):
            entry_name = (
                entry.decode("utf-8")
                if isinstance(entry, bytes)
                else str(entry)
            )
            entry_name = entry_name.rstrip("/")
            full = f"{curr}/{entry_name}"
            if gf.IsDirectory(full):
              queue.append(full)
            else:
              rel = full[len(self.root_path) + 1 :]
              yield rel
        except Exception:
          pass
    else:
      for root, _, files in os.walk(self.root_path):
        for file in files:
          full = os.path.join(root, file)
          rel = full[len(self.root_path) + 1 :]
          yield rel

  def __len__(self) -> int:
    return sum(1 for _ in iter(self))


def torch_safe_save(obj: Any, path: Union[str, Path]) -> None:
  """Saves PyTorch tensors or state dicts to CNS or local path via gfile."""
  p_str = str(path)
  parent = os.path.dirname(p_str)
  if parent:
    gfile_makedirs(parent, exist_ok=True)
  with gfile_open(p_str, "wb") as f:
    torch.save(obj, f)


def torch_safe_load(
    path: Union[str, Path], map_location: Any = "cpu", weights_only: bool = True
) -> Any:
  """Loads PyTorch checkpoint from CNS or local path via gfile with secure weights_only=True default."""
  p_str = str(path)
  with gfile_open(p_str, "rb") as f:
    return torch.load(f, map_location=map_location, weights_only=weights_only)
