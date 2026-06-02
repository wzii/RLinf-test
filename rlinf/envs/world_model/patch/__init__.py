# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Patches for diffsynth-studio's Wan video DiT, installed via the shared Patcher.

These objects are referenced by the Patcher via their dotted paths in
``WanEnv._build_pipeline`` and re-exported here for direct import.
"""

from rlinf.envs.world_model.patch.wan_video_dit import (
    RMSNorm,
    flash_attention,
    rope_apply,
)

__all__ = ["RMSNorm", "flash_attention", "rope_apply"]
