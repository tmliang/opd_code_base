"""Preprocess Vision-OPD-6K (train) and vstar_bench (val) into parquet.

Mirrors ``Vision-OPD/scripts/prepare_data.py``: images are stored as
``{"path": absolute_path}`` (lazy load at training time) so the parquet stays
tiny — no embedded bytes, no HF sharding into ``*_00000.parquet`` shards.

Outputs two independent parquets named after the source dataset:
  ``<out_dir>/Vision-OPD-6K.parquet``  — train
  ``<out_dir>/vstar_bench.parquet``    — val
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


REMOVE_HINT = (
    "Only focus on the objects inside the red bounding box in the image "
    "to answer this question."
)


def _clean_question(problem: str) -> str:
    text = (problem or "").replace("<image>", "").strip()
    text = text.replace(f"\n\n{REMOVE_HINT}", "").replace(REMOVE_HINT, "")
    return text.strip()


def build_train(jsonl: Path, root: Path, limit: int = 0) -> pd.DataFrame:
    rows = []
    with open(jsonl) as f:
        for idx, line in enumerate(f):
            if limit > 0 and idx >= limit:
                break
            ex = json.loads(line)
            student = [{"image": str((root / p).resolve())} for p in ex["images"]]
            teacher = [{"image": str((root / p).resolve())} for p in ex.get("teacher_images", [])]
            answer = ex.get("answer", "")
            problem = ex["problem"]
            rows.append({
                "data_source": "Vision-OPD-6K",
                "prompt": [{"role": "user", "content": problem}],
                "images": student,
                "ability": "visual_question_answering",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": "train",
                    "index": idx,
                    "answer": answer,
                    "question": _clean_question(problem),
                    "teacher_images": teacher,
                },
            })
    return pd.DataFrame(rows)


def build_test(jsonl: Path, root: Path, limit: int = 0) -> pd.DataFrame:
    rows = []
    with open(jsonl) as f:
        for idx, line in enumerate(f):
            if limit > 0 and idx >= limit:
                break
            ex = json.loads(line)
            img = {"image": str((root / ex["image"]).resolve())}
            text = ex["text"]
            if "<image>" not in text:
                text = "<image>\n" + text
            answer = ex["label"]
            rows.append({
                "data_source": "vstar_bench",
                "prompt": [{"role": "user", "content": text}],
                "images": [img],
                "ability": "visual_question_answering",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": "test",
                    "index": idx,
                    "answer": answer,
                    "question": text,
                    "category": ex.get("category"),
                    "question_id": ex.get("question_id"),
                },
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_root", default=os.path.expanduser("~/data/Vision-OPD-6K"))
    p.add_argument("--test_root", default=os.path.expanduser("~/data/vstar_bench"))
    p.add_argument("--out_dir", default=os.path.expanduser("~/data/vision_opd"))
    p.add_argument("--train_limit", type=int, default=0, help="if >0, keep first N train rows")
    p.add_argument("--test_limit", type=int, default=0, help="if >0, keep first N test rows")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = build_train(Path(args.train_root) / "train.jsonl", Path(args.train_root), args.train_limit)
    test = build_test(Path(args.test_root) / "test_questions.jsonl", Path(args.test_root), args.test_limit)
    print(f"train rows={len(train)}  test rows={len(test)}")
    train_path = out / "Vision-OPD-6K.parquet"
    test_path = out / "vstar_bench.parquet"
    train.to_parquet(str(train_path), index=False)
    test.to_parquet(str(test_path), index=False)
    print(f"wrote {train_path}  {test_path}")
