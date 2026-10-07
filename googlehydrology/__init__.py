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
try:
  import torch
  # Disable TorchDynamo / torch.compile to avoid Borglet C++/Inductor sandbox failures
  if hasattr(torch, "_dynamo"):
    torch._dynamo.config.disable = True
except (ImportError, AttributeError, ModuleNotFoundError):
  pass
