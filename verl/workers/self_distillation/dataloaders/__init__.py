# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Built-in teacher dataloader recipes (OPSD / SDPO / Vision-OPD).

Recipes are registered by short name and resolved by
:func:`verl.workers.self_distillation.runtime.load_teacher_dataloader_from_spec`,
which also still accepts a full ``pkg.module:Class`` FQN for user code.
"""

from __future__ import annotations

from typing import Type

from ..teacher_dataloader import TeacherDataloader

_REGISTRY: dict[str, Type[TeacherDataloader]] = {}


def register(name: str):
    """Decorator: register a TeacherDataloader subclass under a short name."""

    def _wrap(cls: Type[TeacherDataloader]):
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"teacher dataloader name {name!r} already registered to {_REGISTRY[name]!r}")
        _REGISTRY[name] = cls
        return cls

    return _wrap


def get(name: str) -> Type[TeacherDataloader] | None:
    return _REGISTRY.get(name)


def names() -> list[str]:
    return sorted(_REGISTRY)


# Import recipe modules so their @register decorators run.
from . import opsd_dataloader as _opsd  # noqa: E402,F401
from . import sdpo_dataloader as _sdpo  # noqa: E402,F401
from . import vision_opd_dataloader as _vision  # noqa: E402,F401
from . import video_opd_dataloader as _video  # noqa: E402,F401
