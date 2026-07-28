# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import contextvars
from typing import (Any, Callable, Dict, Iterator, List, Optional, Tuple,
                    Union)

import torch

_ACTIVE_NCCL_WINDOW_REUSE_DOMAIN_ID: contextvars.ContextVar[
    Optional[int]] = contextvars.ContextVar(
        "active_nccl_window_reuse_domain_id", default=None)
_ACTIVE_NCCL_WINDOW_REUSE_LANE: contextvars.ContextVar[
    Optional[Tuple[int, int]]] = contextvars.ContextVar(
        "active_nccl_window_reuse_lane", default=None)
_NCCL_WINDOW_REUSE_DOMAIN_RETIRE_CALLBACKS: List[Callable[[int],
                                                          None]] = []


def get_active_nccl_window_reuse_domain_id() -> Optional[int]:
    """Return the domain whose registered-window capacity is being prepared."""
    return _ACTIVE_NCCL_WINDOW_REUSE_DOMAIN_ID.get()


def register_nccl_window_reuse_domain_retire_callback(
        callback: Callable[[int], None]) -> None:
    """Register host-only cleanup for state keyed by a native domain handle."""
    _NCCL_WINDOW_REUSE_DOMAIN_RETIRE_CALLBACKS.append(callback)


class NCCLWindowReuseDomain:
    """Own the registered-window arena shared by serial CUDA graph variants.

    One native domain is created per ``(device, replay stream)`` lane. Captures
    on the same lane may share registered scratch when their replays are serial;
    different lanes remain disjoint. Capture bookkeeping runs only while a
    graph is being recorded; replay stays a direct ``CUDAGraph.replay()`` with
    no Python or allocator-side checks.

    Owners must call ``quiesce()``, reset every graph captured with the domain,
    and then call ``close()``. A closed instance can be reused for a later
    capture generation.
    """

    def __init__(self, *, borrow_active: bool = False) -> None:
        self._handles: Dict[Tuple[int, int], int] = {}
        self._quiesced_handles: set[int] = set()
        self._borrow_active = borrow_active

    @staticmethod
    def _current_lane(
        device: Optional[Union[int, torch.device]] = None,
    ) -> Optional[Tuple[int, int]]:
        if not torch.cuda.is_available():
            return None

        if device is None:
            device_index = torch.cuda.current_device()
        elif isinstance(device, int):
            device_index = device
        else:
            cuda_device = torch.device(device)
            if cuda_device.type != "cuda":
                raise ValueError(
                    "NCCL window reuse domains require a CUDA device")
            device_index = (torch.cuda.current_device()
                            if cuda_device.index is None else cuda_device.index)

        replay_stream = torch.cuda.current_stream(device_index).cuda_stream
        return device_index, replay_stream

    def _ensure(
        self,
        device: Optional[Union[int, torch.device]] = None,
    ) -> Optional[int]:
        lane = self._current_lane(device)
        if lane is None:
            return None
        if lane in self._handles:
            handle = self._handles[lane]
            if handle in self._quiesced_handles:
                raise RuntimeError(
                    "Cannot prepare a quiesced NCCL window reuse domain")
            return handle

        create_op = getattr(torch.ops.trtllm,
                            "_create_nccl_window_reuse_domain", None)
        if create_op is None:
            return None

        device_index, _ = lane
        handle = int(create_op(device_index))
        if handle == 0:
            return None
        self._handles[lane] = handle
        return handle

    @contextlib.contextmanager
    def prepare(self) -> Iterator[Optional[int]]:
        """Expose this lane's identity while eager setup allocates capacity."""
        lane = self._current_lane()
        active_handle = _ACTIVE_NCCL_WINDOW_REUSE_DOMAIN_ID.get()
        if active_handle is not None:
            active_lane = _ACTIVE_NCCL_WINDOW_REUSE_LANE.get()
            if lane != active_lane:
                raise RuntimeError(
                    "Cannot reuse an NCCL window domain on a different CUDA "
                    f"stream: active lane={active_lane}, current lane={lane}")
            if active_handle in self._handles.values():
                yield active_handle
                return
            if self._borrow_active and not self._handles:
                # Short-lived helpers such as the autotuner may capture
                # temporary graphs in their caller's serial domain. They must
                # reset those graphs before returning; this instance neither
                # records nor closes the borrowed handle.
                yield active_handle
                return
            raise RuntimeError(
                "Cannot nest different NCCL window reuse domains")

        handle = self._ensure()
        begin_op = getattr(torch.ops.trtllm,
                           "_begin_nccl_window_preparation", None)
        end_op = getattr(torch.ops.trtllm,
                         "_end_nccl_window_preparation", None)
        if handle is not None and begin_op is not None and end_op is not None:
            begin_op(handle)
        token = _ACTIVE_NCCL_WINDOW_REUSE_DOMAIN_ID.set(handle)
        lane_token = _ACTIVE_NCCL_WINDOW_REUSE_LANE.set(lane)
        try:
            yield handle
        finally:
            _ACTIVE_NCCL_WINDOW_REUSE_LANE.reset(lane_token)
            _ACTIVE_NCCL_WINDOW_REUSE_DOMAIN_ID.reset(token)
            if handle is not None and begin_op is not None and end_op is not None:
                end_op(handle)

    @contextlib.contextmanager
    def capture(
        self,
        graph: torch.cuda.CUDAGraph,
        *,
        pool: Any = None,
    ) -> Iterator[None]:
        """Capture ``graph`` and bind any used NCCL windows to this domain."""
        with self.prepare() as handle:
            with torch.cuda.graph(graph, pool=pool):
                capture_id = None
                begin_op = getattr(torch.ops.trtllm,
                                   "_begin_nccl_window_capture", None)
                end_op = getattr(torch.ops.trtllm,
                                 "_end_nccl_window_capture", None)
                if (handle is not None and begin_op is not None
                        and end_op is not None):
                    capture_id = int(begin_op(handle))
                try:
                    yield
                finally:
                    if capture_id is not None:
                        end_op(capture_id)

    def quiesce(self) -> None:
        """Reject new native work before resetting this owner's graphs."""
        quiesce_op = getattr(torch.ops.trtllm,
                             "_quiesce_nccl_window_reuse_domain", None)
        for handle in self._handles.values():
            if handle in self._quiesced_handles:
                continue
            if quiesce_op is not None:
                quiesce_op(handle)
            self._quiesced_handles.add(handle)

    def close(self) -> None:
        """Release the domain after all of its captured graphs were reset."""
        if not self._handles:
            return
        self.quiesce()
        close_op = getattr(torch.ops.trtllm,
                           "_close_nccl_window_reuse_domain", None)
        for lane, handle in list(self._handles.items()):
            if close_op is not None:
                close_op(handle)
            for callback in tuple(
                    _NCCL_WINDOW_REUSE_DOMAIN_RETIRE_CALLBACKS):
                callback(handle)
            del self._handles[lane]
            self._quiesced_handles.discard(handle)
