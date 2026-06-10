# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Video-OPD teacher dataloader (identity prompt + time-reference-centred frame resampling).

Student rollout keeps the standard pipeline (uniform frame sampling done by the
dataset / qwen_vl_utils). The teacher re-decodes the *same* video with a
non-uniform frame schedule centred on the answer's time interval
(``extra_info["time_reference"]``), so the teacher's distribution is computed
on frames that actually contain the evidence — that perception gap is the
distillation signal. No text is changed.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from verl.workers.self_distillation import OfflineTeacherDataloader, TeacherSample

from . import register


def _strip_file_uri(path: str) -> str:
    return path[len("file://") :] if path.startswith("file://") else path


@register("video_opd")
class video_opd_dataloader(OfflineTeacherDataloader):  # noqa: N801 (lowercase per repo convention)
    """Teacher = same text prompt, video re-sampled non-uniformly around the
    answer's time interval.

    Frame schedule: a ``focus_ratio`` fraction of the frame budget is placed
    uniformly inside ``[start - context_margin_sec, end + context_margin_sec]``
    (clamped to the video), the rest uniformly over the whole duration; the
    union is sorted by timestamp. The frame budget defaults to the student's
    frame count, so teacher and student see the same number of video tokens.

    Expected ``extra_info`` fields:

    - ``time_reference`` (configurable via ``time_reference_field``): the
      answer interval in seconds — ``[start, end]`` / ``(start, end)`` /
      ``{"start": s, "end": e}`` / ``{"start_time": s, "end_time": e}``.
      A single number is treated as a point interval.
    - the source video: taken from ``extra_info[video_path_field]`` when
      ``video_path_field`` is set, otherwise from the first
      ``{"type": "video", "video": <path>}`` segment in the messages.

    YAML knobs (``self_distill.dataloader_kwargs``):

    - ``time_reference_field`` (str, default ``"time_reference"``).
    - ``video_path_field`` (str, default ``None`` → read from messages).
    - ``num_frames`` (int, default ``None`` → match the student's sampled
      frame count, falling back to 16 when it cannot be inferred).
    - ``focus_ratio`` (float in (0, 1], default 0.6): fraction of frames
      inside the (margin-widened) reference interval.
    - ``context_margin_sec`` (float, default 2.0): widen the interval by this
      many seconds on each side before placing focus frames.

    Notes:

    - Temporal MROPE / ``second_per_grid_ts`` assumes near-uniform frame
      spacing; with non-uniform sampling the teacher's temporal rope is
      approximate. Empirically irrelevant for distillation logprobs, but keep
      it in mind when comparing against the student forward.
    - Decoding prefers ``decord``, falling back to ``torchvision.io``.
    """

    def __init__(self, *, tokenizer, processor=None, **kwargs):
        super().__init__(tokenizer=tokenizer, processor=processor, **kwargs)
        self._time_field = str(self.config.get("time_reference_field", "time_reference"))
        self._video_path_field = self.config.get("video_path_field")
        nf = self.config.get("num_frames")
        self._num_frames = int(nf) if nf else None
        self._focus_ratio = float(self.config.get("focus_ratio", 0.6))
        if not (0.0 < self._focus_ratio <= 1.0):
            raise ValueError(f"focus_ratio must be in (0, 1], got {self._focus_ratio}")
        self._context_margin = float(self.config.get("context_margin_sec", 2.0))

    # ------------------------------------------------------------ parsing

    @staticmethod
    def _parse_time_reference(value: Any) -> Optional[tuple[float, float]]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            t = float(value)
            return (t, t)
        if isinstance(value, dict):
            start = value.get("start", value.get("start_time"))
            end = value.get("end", value.get("end_time"))
            if start is None or end is None:
                return None
            return (float(start), float(end))
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return (float(value[0]), float(value[1]))
        return None

    def _find_video_path(self, messages: list[dict], extra_info: dict) -> Optional[str]:
        if self._video_path_field:
            p = extra_info.get(self._video_path_field)
            return _strip_file_uri(str(p)) if p else None
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "video":
                    v = item.get("video")
                    if isinstance(v, str):
                        return _strip_file_uri(v)
        return None

    @staticmethod
    def _infer_student_num_frames(multi_modal_data: Optional[dict]) -> Optional[int]:
        videos = (multi_modal_data or {}).get("videos")
        if not videos:
            return None
        v = videos[0]
        if isinstance(v, tuple):  # (frames_tensor, metadata)
            v = v[0]
        if isinstance(v, torch.Tensor) and v.dim() == 4:
            return int(v.shape[0])
        if isinstance(v, (list, tuple)):
            return len(v)
        return None

    # ------------------------------------------------------------ sampling

    def _build_frame_indices(
        self, *, total_frames: int, fps: float, start: float, end: float, n: int
    ) -> list[int]:
        """Non-uniform frame indices: dense inside the (widened) interval, sparse outside."""
        duration = total_frames / fps if fps > 0 else 0.0
        if duration <= 0 or total_frames <= 0:
            return []
        n = max(1, min(n, total_frames))

        s = max(0.0, min(start, duration))
        e = max(s, min(end, duration))
        fs = max(0.0, s - self._context_margin)
        fe = min(duration, e + self._context_margin)

        n_focus = max(1, min(n, round(n * self._focus_ratio)))
        n_rest = n - n_focus

        ts = torch.linspace(fs, fe, n_focus).tolist()
        if n_rest > 0:
            ts += torch.linspace(0.0, duration, n_rest).tolist()

        existing = {min(total_frames - 1, max(0, round(t * fps))) for t in ts}
        # Deduplication can shrink the set (short clips / tight intervals or
        # focus/uniform schedules colliding); top up with unused frames closest
        # to the center of the focus window.
        if len(existing) < n:
            center = (fs + fe) / 2.0 * fps
            for c in sorted(range(total_frames), key=lambda i: abs(i - center)):
                if c not in existing:
                    existing.add(c)
                    if len(existing) >= n:
                        break
        return sorted(existing)

    # ------------------------------------------------------------ decoding

    @staticmethod
    def _decode_video(path: str, indices: list[int]) -> torch.Tensor:
        """Return frames as a uint8 tensor of shape (n, 3, H, W)."""
        try:
            from decord import VideoReader, cpu

            vr = VideoReader(path, ctx=cpu(0))
            frames = torch.from_numpy(vr.get_batch(indices).asnumpy())  # (n, H, W, 3)
            return frames.permute(0, 3, 1, 2).contiguous()
        except ImportError:
            pass
        import torchvision.io

        video, _, _ = torchvision.io.read_video(path, output_format="TCHW", pts_unit="sec")
        idx = torch.tensor([min(i, video.shape[0] - 1) for i in indices], dtype=torch.long)
        return video.index_select(0, idx).contiguous()

    @staticmethod
    def _probe_video(path: str) -> tuple[int, float]:
        """Return (total_frames, fps)."""
        try:
            from decord import VideoReader, cpu

            vr = VideoReader(path, ctx=cpu(0))
            return len(vr), float(vr.get_avg_fps())
        except ImportError:
            pass
        import torchvision.io

        video, _, info = torchvision.io.read_video(path, output_format="TCHW", pts_unit="sec")
        return int(video.shape[0]), float(info.get("video_fps", 0.0))

    @staticmethod
    def _build_metadata(*, total_frames: int, fps: float, indices: list[int]):
        meta = {
            "total_num_frames": int(total_frames),
            "fps": float(fps),
            "duration": (total_frames / fps) if fps > 0 else 0.0,
            "frames_indices": list(indices),
            "video_backend": "decord",
        }
        try:
            from transformers.video_utils import VideoMetadata

            return VideoMetadata(**meta)
        except (ImportError, TypeError):
            return meta

    # ------------------------------------------------------------ build_one

    def build_one(self, *, prompt_messages, prompt_text, multi_modal_data, extra_info):
        if prompt_messages is None:
            return TeacherSample(messages=[], skip=True)

        ei = extra_info or {}
        interval = self._parse_time_reference(ei.get(self._time_field))
        if interval is None:
            return TeacherSample(messages=[], skip=True)

        video_path = self._find_video_path(prompt_messages, ei)
        if not video_path:
            return TeacherSample(messages=[], skip=True)

        n = self._num_frames or self._infer_student_num_frames(multi_modal_data) or 16

        try:
            total_frames, fps = self._probe_video(video_path)
            indices = self._build_frame_indices(
                total_frames=total_frames, fps=fps, start=interval[0], end=interval[1], n=n
            )
            if not indices:
                return TeacherSample(messages=[], skip=True)
            frames = self._decode_video(video_path, indices)
        except Exception as exc:  # noqa: BLE001 — corrupt/missing video: mask, don't crash the step
            import logging

            logging.getLogger(__name__).warning(
                "video_opd: failed to resample %s (%s); skipping sample.", video_path, exc
            )
            return TeacherSample(messages=[], skip=True)

        metadata = self._build_metadata(total_frames=total_frames, fps=fps, indices=indices)

        # Text prompt is byte-identical to the student's; only the video frames
        # change. The (frames, metadata) tuple goes through
        # ``build_multimodal_processor_inputs`` → ``do_sample_frames=False``,
        # so the processor consumes exactly these frames.
        mm = dict(multi_modal_data or {})
        mm["videos"] = [(frames, metadata)]
        return TeacherSample(messages=list(prompt_messages), multi_modal_data=mm)
