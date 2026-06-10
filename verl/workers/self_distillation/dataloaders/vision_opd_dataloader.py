# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Vision-OPD teacher dataloader (identity prompt + bbox-image swap)."""

from __future__ import annotations

import json
import os
import threading

from verl.workers.self_distillation import OfflineTeacherDataloader, TeacherSample

from . import register

# Opt-in dump for correctness verification.
#   TEACHER_DUMP_DIR=/path  -> enable
#   TEACHER_DUMP_MAX=N      -> per-process cap (default 16)
_DUMP_DIR = os.environ.get("TEACHER_DUMP_DIR") or None
_DUMP_MAX = int(os.environ.get("TEACHER_DUMP_MAX", "16"))
_DUMP_LOCK = threading.Lock()
_DUMP_COUNTER = {"n": 0}


@register("vision_opd")
class VisionOPD_dataloader(OfflineTeacherDataloader):  # noqa: N801 (mixed case per user request)
    """Vision-OPD recipe: faithful reproduction of yuanqianhao/Vision-OPD.

    Teacher prompt is byte-for-byte identical to the student prompt; only the
    image pixels are swapped to the teacher-side (``bbox_images``) variant.
    The teacher's distribution on the bbox-highlighted image is what supplies
    the perception signal — no extra text hint is injected.

    Mirrors ``RayPPOTrainer._swap_images_in_messages`` /
    ``_extract_images_from_messages`` from the original codebase: it walks the
    chat messages and replaces each ``{"type": "image", ...}`` entry's pixels
    with the teacher image at the same offset. Sources for teacher images
    (in order of preference): ``extra_info["teacher_images"]``,
    ``extra_info["bbox_images"]``.
    """

    @staticmethod
    def _normalize(image):
        from io import BytesIO

        from PIL import Image

        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        if isinstance(image, dict):
            if "image" in image:
                return VisionOPD_dataloader._normalize(image["image"])
            if image.get("bytes") is not None:
                return Image.open(BytesIO(image["bytes"])).convert("RGB")
            if image.get("path"):
                return Image.open(image["path"]).convert("RGB")
        raise TypeError(f"Unsupported teacher image type: {type(image)}")

    def build_one(self, *, prompt_messages, prompt_text, multi_modal_data, extra_info):
        if prompt_messages is None:
            return TeacherSample(messages=[], skip=True)

        ei = extra_info or {}
        raw = ei.get("teacher_images") or ei.get("bbox_images")
        if not raw:
            return TeacherSample(messages=[], skip=True)
        teacher_pils = [self._normalize(it) for it in raw if it is not None]
        if not teacher_pils:
            return TeacherSample(messages=[], skip=True)

        # Structure-copy the messages (dicts/lists one level deep) instead of
        # ``copy.deepcopy``: the student messages inline PIL images whose pixel
        # buffers we are about to replace anyway — deep-copying them is pure
        # waste (and significant for rollout.n > 1).
        messages = [
            {
                **msg,
                "content": (
                    [dict(it) if isinstance(it, dict) else it for it in msg["content"]]
                    if isinstance(msg.get("content"), list)
                    else msg.get("content")
                ),
            }
            for msg in prompt_messages
        ]
        image_offset = 0
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    if image_offset >= len(teacher_pils):
                        return TeacherSample(messages=[], skip=True)
                    item["image"] = teacher_pils[image_offset]
                    image_offset += 1
        if image_offset != len(teacher_pils):
            # More teacher images than image slots in the messages — the
            # processor would see mismatched text placeholders vs pixels.
            return TeacherSample(messages=[], skip=True)

        mm = dict(multi_modal_data or {})
        # Even when messages carry inlined images, runtime._tokenize_messages
        # feeds the processor from multi_modal_data["images"]; swap it too.
        mm["images"] = teacher_pils

        if _DUMP_DIR is not None:
            self._maybe_dump(
                student_messages=prompt_messages,
                teacher_messages=messages,
                prompt_text=prompt_text,
                student_images=(multi_modal_data or {}).get("images"),
                teacher_images=teacher_pils,
                extra_info=ei,
            )

        return TeacherSample(messages=messages, multi_modal_data=mm)

    @staticmethod
    def _maybe_dump(
        *, student_messages, teacher_messages, prompt_text,
        student_images, teacher_images, extra_info,
    ):
        with _DUMP_LOCK:
            if _DUMP_COUNTER["n"] >= _DUMP_MAX:
                return
            idx = _DUMP_COUNTER["n"]
            _DUMP_COUNTER["n"] += 1
        sample_dir = os.path.join(_DUMP_DIR, f"pid{os.getpid()}_sample{idx:04d}")
        os.makedirs(sample_dir, exist_ok=True)

        def _strip(messages):
            out = []
            for m in messages or []:
                content = m.get("content")
                if isinstance(content, list):
                    new_content = []
                    img_k = 0
                    for it in content:
                        if isinstance(it, dict) and it.get("type") == "image":
                            new_content.append({"type": "image", "_ref": f"image_{img_k}.png"})
                            img_k += 1
                        else:
                            new_content.append(it)
                    out.append({**m, "content": new_content})
                else:
                    out.append(m)
            return out

        with open(os.path.join(sample_dir, "student_messages.json"), "w") as f:
            json.dump(_strip(student_messages), f, ensure_ascii=False, indent=2, default=str)
        with open(os.path.join(sample_dir, "teacher_messages.json"), "w") as f:
            json.dump(_strip(teacher_messages), f, ensure_ascii=False, indent=2, default=str)
        with open(os.path.join(sample_dir, "student_prompt.txt"), "w") as f:
            f.write(prompt_text or "")
        with open(os.path.join(sample_dir, "extra_info.json"), "w") as f:
            json.dump(
                {k: v for k, v in (extra_info or {}).items()
                 if k not in ("teacher_images", "bbox_images", "images")},
                f, ensure_ascii=False, indent=2, default=str,
            )

        try:
            for k, img in enumerate(student_images or []):
                pil = VisionOPD_dataloader._normalize(img)
                pil.save(os.path.join(sample_dir, f"student_image_{k}.png"))
            for k, img in enumerate(teacher_images or []):
                img.save(os.path.join(sample_dir, f"teacher_image_{k}.png"))
        except Exception as e:
            with open(os.path.join(sample_dir, "image_dump_error.txt"), "w") as f:
                f.write(repr(e))
