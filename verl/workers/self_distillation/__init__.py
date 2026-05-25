# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""On-Policy Self-Distillation (OPSD) plug-in for verl.

OPSD reuses verl's OPD loss kernel verbatim (``verl.trainer.distillation``).
The only thing it changes is *who* produces ``teacher_logprobs`` / ``teacher_ids``:
instead of an external vLLM teacher cluster, a colocated reference policy is
forwarded over a **rewritten prompt** (same response). The prompt rewrite is
defined by a user-supplied teacher dataloader.

Public surface:
* :class:`TeacherSample` — return type of a teacher dataloader.
* :class:`OfflineTeacherDataloader` — base class; depends only on the original sample.
* :class:`OnlineTeacherDataloader` — base class; depends on the student rollout.
* :class:`BatchView` — read-only batch helper passed to online dataloaders.
* :class:`SelfDistillationRuntime` — internal glue (per-sample tokenization + concat).
* :func:`load_teacher_dataloader_from_spec` — instantiate from config.
"""

from .teacher_dataloader import (
    BatchView,
    OfflineTeacherDataloader,
    OnlineTeacherDataloader,
    TeacherDataloader,
    TeacherSample,
)
from .runtime import SelfDistillationRuntime, load_teacher_dataloader_from_spec

__all__ = [
    "BatchView",
    "OfflineTeacherDataloader",
    "OnlineTeacherDataloader",
    "TeacherDataloader",
    "TeacherSample",
    "SelfDistillationRuntime",
    "load_teacher_dataloader_from_spec",
]
