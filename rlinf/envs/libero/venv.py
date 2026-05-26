# Copyright 2025 The RLinf Authors.
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

import multiprocessing
import os
import time
import warnings
from multiprocessing import connection
from typing import Any, Callable, Optional, Union

import gym
import numpy as np

from rlinf.envs.libero.utils import get_libero_type
from rlinf.envs.venv import (
    BaseVectorEnv,
    CloudpickleWrapper,
    EnvWorker,
    ShArray,
    SubprocEnvWorker,
    SubprocVectorEnv,
    _setup_buf,
)

# ---------------------------------------------------------------------------
# Dynamic Module Import Logic for Libero Pro / Plus
# ---------------------------------------------------------------------------
libero_type = get_libero_type()

if libero_type == "pro":
    try:
        from liberopro.liberopro.envs import OffScreenRenderEnv
    except ImportError as e:
        print(
            f"[Venv] Warning: LIBERO_TYPE=pro but import failed ({e}). Falling back to standard libero..."
        )
        from libero.libero.envs import OffScreenRenderEnv

elif libero_type == "plus":
    try:
        from liberoplus.liberoplus.envs import OffScreenRenderEnv
    except ImportError as e:
        print(
            f"[Venv] Warning: LIBERO_TYPE=plus but import failed ({e}). Falling back to standard libero..."
        )
        from libero.libero.envs import OffScreenRenderEnv

else:
    try:
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError:
        try:
            from liberopro.liberopro.envs import OffScreenRenderEnv
        except ImportError:
            try:
                from liberoplus.liberoplus.envs import OffScreenRenderEnv
            except ImportError:
                raise ImportError(
                    "Could not import OffScreenRenderEnv from libero, liberopro, or liberoplus."
                )


gym_old_venv_step_type = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
gym_new_venv_step_type = tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]
warnings.simplefilter("once", DeprecationWarning)


def _worker(
    parent: connection.Connection,
    p: connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    obs_bufs: Optional[Union[dict, tuple, ShArray]] = None,
    extra_env: Optional[dict] = None,
) -> None:
    # Apply extra env vars before any MuJoCo / OpenGL initialisation.
    # With multiprocessing.spawn the child process starts fresh and may not
    # inherit MUJOCO_GL / PYOPENGL_PLATFORM from the parent's runtime_env.
    import os as _os
    # Unconditionally limit OpenMP threads in each spawned env-worker subprocess.
    # Without this, MuJoCo spawns O(num_cpus) threads per process.  With 50+
    # concurrent workers that exhausts virtual-address space for thread stacks,
    # causing RuntimeError inside mujoco.MjModel.from_xml_string.
    # The RLINF_ENV_OMP_THREADS env var lets callers override if needed.
    _omp = _os.environ.get('RLINF_ENV_OMP_THREADS', '1')
    _os.environ['OMP_NUM_THREADS'] = _omp
    _os.environ['MKL_NUM_THREADS'] = _omp
    # Ensure EGL rendering is configured before any MuJoCo/robosuite import.
    # These may not be set in spawned subprocesses even if parent process had them.
    _os.environ.setdefault('MUJOCO_GL', 'egl')
    _os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    if extra_env:
        for k, v in extra_env.items():
            _os.environ[k] = v

    def _encode_obs(
        obs: Union[dict, tuple, np.ndarray], buffer: Union[dict, tuple, ShArray]
    ) -> None:
        if isinstance(obs, np.ndarray) and isinstance(buffer, ShArray):
            buffer.save(obs)
        elif isinstance(obs, tuple) and isinstance(buffer, tuple):
            for o, b in zip(obs, buffer):
                _encode_obs(o, b)
        elif isinstance(obs, dict) and isinstance(buffer, dict):
            for k in obs.keys():
                _encode_obs(obs[k], buffer[k])
        return None

    parent.close()
    env = env_fn_wrapper.data()
    try:
        while True:
            try:
                cmd, data = p.recv()
            except EOFError:  # the pipe has been closed
                p.close()
                break
            if cmd == "step":
                env_return = env.step(data)
                if obs_bufs is not None:
                    _encode_obs(env_return[0], obs_bufs)
                    env_return = (None, *env_return[1:])
                p.send(env_return)
            elif cmd == "reset":
                retval = env.reset(**data)
                reset_returns_info = (
                    isinstance(retval, (tuple, list))
                    and len(retval) == 2
                    and isinstance(retval[1], dict)
                )
                if reset_returns_info:
                    obs, info = retval
                else:
                    obs = retval
                if obs_bufs is not None:
                    _encode_obs(obs, obs_bufs)
                    obs = None
                if reset_returns_info:
                    p.send((obs, info))
                else:
                    p.send(obs)
            elif cmd == "close":
                p.send(env.close())
                p.close()
                break
            elif cmd == "render":
                p.send(env.render(**data) if hasattr(env, "render") else None)
            elif cmd == "seed":
                if hasattr(env, "seed"):
                    p.send(env.seed(data))
                else:
                    env.reset(seed=data)
                    p.send(None)
            elif cmd == "getattr":
                p.send(getattr(env, data) if hasattr(env, data) else None)
            elif cmd == "setattr":
                setattr(env.unwrapped, data["key"], data["value"])
            elif cmd == "check_success":
                p.send(env.check_success())
            elif cmd == "get_segmentation_of_interest":
                p.send(env.get_segmentation_of_interest(data))
            elif cmd == "get_sim_state":
                p.send(env.get_sim_state())
            elif cmd == "set_init_state":
                obs = env.set_init_state(data)
                p.send(obs)
            elif cmd == "reconfigure":
                env.close()
                seed = data.pop("seed")
                env = OffScreenRenderEnv(**data)
                env.seed(seed)
                p.send(None)
            else:
                p.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        p.close()


class ReconfigureSubprocEnvWorker(SubprocEnvWorker):
    # Environment variables that must be forwarded to the spawned subprocess
    # because multiprocessing.spawn on Linux may not inherit them from the
    # Ray worker's runtime_env.  MUJOCO_GL and PYOPENGL_PLATFORM control
    # headless / EGL rendering; missing them causes MuJoCo XML loading errors.
    # OMP_NUM_THREADS is forwarded so a deliberate user override propagates.
    _FORWARD_ENV_VARS = ("MUJOCO_GL", "PYOPENGL_PLATFORM", "MUJOCO_EGL_DEVICE_ID",
                         "EGL_DEVICE_ID", "MUJOCO_EGL_DEVICE_INDEX", "OMP_NUM_THREADS")

    def __init__(self, env_fn: Callable[[], gym.Env], share_memory: bool = False):
        import os as _os
        ctx = multiprocessing.get_context("spawn")
        self.parent_remote, self.child_remote = ctx.Pipe()
        self.share_memory = share_memory
        self.buffer: Optional[Union[dict, tuple, ShArray]] = None
        if self.share_memory:
            dummy = env_fn()
            obs_space = dummy.observation_space
            dummy.close()
            del dummy
            self.buffer = _setup_buf(obs_space)
        # Forward rendering-related env vars so spawned workers can
        # initialise EGL/OSMesa before MuJoCo is first imported.
        extra_env = {k: v for k in self._FORWARD_ENV_VARS
                     if (v := _os.environ.get(k)) is not None}
        args = (
            self.parent_remote,
            self.child_remote,
            CloudpickleWrapper(env_fn),
            self.buffer,
            extra_env,
        )
        self.process = ctx.Process(target=_worker, args=args, daemon=True)
        self.process.start()
        self.child_remote.close()
        EnvWorker.__init__(self, env_fn)

    def reconfigure_env_fn(self, env_fn_param):
        self.parent_remote.send(["reconfigure", env_fn_param])
        return self.parent_remote.recv()


class ReconfigureSubprocEnv(SubprocVectorEnv):
    # Spawn environment workers in batches to avoid exhausting OS resources
    # (EGL contexts, virtual memory for thread stacks) when creating many
    # parallel subprocess workers simultaneously.
    #
    # Defaults: batch of 10, 2 s gap.  Override via env vars:
    #   RLINF_ENV_SPAWN_BATCH  – workers per batch  (0 = all at once)
    #   RLINF_ENV_SPAWN_DELAY  – seconds between batch starts
    _DEFAULT_SPAWN_BATCH = 10
    _DEFAULT_SPAWN_DELAY = 2.0

    def __init__(self, env_fns: list[Callable[[], gym.Env]], **kwargs: Any) -> None:
        batch_size = int(os.environ.get("RLINF_ENV_SPAWN_BATCH",
                                        self._DEFAULT_SPAWN_BATCH))
        batch_delay = float(os.environ.get("RLINF_ENV_SPAWN_DELAY",
                                           self._DEFAULT_SPAWN_DELAY))
        if batch_size <= 0:
            batch_size = len(env_fns)  # effectively disabled

        _counter = [0]

        def worker_fn(fn: Callable[[], gym.Env]) -> ReconfigureSubprocEnvWorker:
            idx = _counter[0]
            _counter[0] += 1
            # Pause before starting each new batch (except the very first).
            if idx > 0 and idx % batch_size == 0:
                time.sleep(batch_delay)
            return ReconfigureSubprocEnvWorker(fn, share_memory=False)

        BaseVectorEnv.__init__(self, env_fns, worker_fn, **kwargs)

    def reconfigure_env_fns(self, env_fns, id=None):
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        for j, i in enumerate(id):
            self.workers[i].reconfigure_env_fn(env_fns[j])
