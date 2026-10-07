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

"""Helper to safely acquire gfile in both Google3 and standalone environments."""

import io
import os
import glob
import subprocess
import tempfile


class FileUtilGFileStream(io.BytesIO):

  def __init__(self, path, mode='r'):
    self.path = str(path)
    self.mode = mode
    if 'r' in mode:
      data = subprocess.check_output(['fileutil', 'cat', self.path])
      super().__init__(data)
    else:
      super().__init__()

  def close(self):
    if 'w' in self.mode or 'a' in self.mode:
      self.seek(0)
      content = self.read()
      with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
      subprocess.check_call(['fileutil', 'cp', '-f', tmp_path, self.path])
      os.remove(tmp_path)
    super().close()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()


class FileUtilGModule:

  @staticmethod
  def Exists(path):
    p = str(path)
    if not p.startswith('/cns/'):
      return os.path.exists(p)
    res = subprocess.run(
        ['fileutil', 'stat', p],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return res.returncode == 0

  @staticmethod
  def GFile(path, mode='r'):
    p = str(path)
    if not p.startswith('/cns/'):
      return open(p, mode)
    return FileUtilGFileStream(p, mode)

  @staticmethod
  def MakeDirs(path):
    p = str(path)
    if not p.startswith('/cns/'):
      os.makedirs(p, exist_ok=True)
    else:
      subprocess.run(['fileutil', 'mkdir', '-p', p], check=False)

  @staticmethod
  def Glob(pattern):
    p = str(pattern)
    if not p.startswith('/cns/'):
      return glob.glob(p)
    try:
      if '*' in p or '?' in p or '[' in p:
        parts = p.split('/')
        base_parts = []
        in_wildcard = False
        for pt in parts:
          if any(c in pt for c in ['*', '?', '[']):
            in_wildcard = True
          if not in_wildcard:
            base_parts.append(pt)
        base_dir = '/'.join(base_parts) if base_parts else '/'
        out = subprocess.check_output(['fileutil', 'ls', base_dir]).decode('utf-8')
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        import fnmatch
        return [line for line in lines if fnmatch.fnmatch(line, p)]
      else:
        out = subprocess.check_output(['fileutil', 'ls', p]).decode('utf-8')
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception:
      return []

  @staticmethod
  def Copy(src, dst, overwrite=True):
    src_s = str(src)
    dst_s = str(dst)
    if not src_s.startswith('/cns/') and not dst_s.startswith('/cns/'):
      import shutil

      shutil.copyfile(src_s, dst_s)
    else:
      cmd = ['fileutil', 'cp']
      if overwrite:
        cmd.append('-f')
      cmd.extend([src_s, dst_s])
      subprocess.check_call(cmd)


def get_gfile():
  try:
    from google3.pyglib import gfile

    return gfile
  except ImportError:
    try:
      from pyglib import gfile

      return gfile
    except ImportError:
      return FileUtilGModule

