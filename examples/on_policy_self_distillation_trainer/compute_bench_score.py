#!/usr/bin/env python3
"""Compute benchmark score from verl validation JSONL dumps."""

from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from statistics import mean


DEFAULT_PATH = Path("outputs/sdpo_gsm8k/eval_gt_opsd_sdpo_qwen35_2b/validation/0.jsonl")
METRIC_PRIORITY = ("acc", "score", "reward")


def resolve_input(path: Path) -> Path:
    if path.is_dir():
        step_files = sorted(
            (p for p in path.glob("*.jsonl") if p.stem.isdigit()),
            key=lambda p: int(p.stem),
        )
        if not step_files:
            raise SystemExit(f"No step JSONL files like 0.jsonl found in directory: {path}")
        return step_files[-1]
    return path


def load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"JSONL file does not exist: {path}")

    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    if not rows:
        raise SystemExit(f"No rows found in JSONL file: {path}")
    return rows


def pick_metric(rows: list[dict], requested_metric: str | None) -> str:
    if requested_metric:
        if not any(requested_metric in row for row in rows):
            raise SystemExit(f"Metric '{requested_metric}' was not found in any row")
        return requested_metric

    for metric in METRIC_PRIORITY:
        if any(metric in row for row in rows):
            return metric
    raise SystemExit(f"No supported metric found. Tried: {', '.join(METRIC_PRIORITY)}")


def numeric_values(rows: list[dict], metric: str) -> list[float]:
    values = []
    for idx, row in enumerate(rows):
        if metric not in row or row[metric] is None:
            continue
        value = row[metric]
        if isinstance(value, bool):
            value = float(value)
        if not isinstance(value, (int, float)):
            raise SystemExit(f"Metric '{metric}' is not numeric at row {idx}: {value!r}")
        values.append(float(value))
    if not values:
        raise SystemExit(f"No numeric values found for metric '{metric}'")
    return values


def normalize_answer(answer: object) -> str:
    text = "" if answer is None else str(answer)
    if "####" in text:
        text = text.split("####")[-1]
    text = text.strip()
    text = text.replace(",", "").replace("$", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\s+", "", text)
    return text.strip()


def extract_last_boxed_content(text: str) -> str | None:
    marker = "\\boxed"
    start = text.rfind(marker)
    if start < 0:
        return None

    pos = start + len(marker)
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos >= len(text):
        return None

    if text[pos] != "{":
        return text[pos:].split("$")[0].strip() or None

    depth = 0
    content_start = pos + 1
    for idx in range(pos, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:idx]
    return None


def extract_boxed_answer(output: object) -> str | None:
    content = extract_last_boxed_content("" if output is None else str(output))
    if content is None:
        return None
    return normalize_answer(content)


def parse_numeric(text: str) -> Decimal | None:
    try:
        return Decimal(text)
    except InvalidOperation:
        pass

    frac_match = re.fullmatch(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", text)
    if frac_match:
        try:
            frac = Fraction(frac_match.group(1)) / Fraction(frac_match.group(2))
            return Decimal(frac.numerator) / Decimal(frac.denominator)
        except (ValueError, ZeroDivisionError, InvalidOperation):
            return None

    return None


def answers_equal(pred: str, target: str) -> bool:
    pred_norm = normalize_answer(pred)
    target_norm = normalize_answer(target)
    if pred_norm == target_norm:
        return True

    pred_num = parse_numeric(pred_norm)
    target_num = parse_numeric(target_norm)
    if pred_num is not None and target_num is not None:
        return pred_num == target_num

    return False


def boxed_accuracy(rows: list[dict], output_key: str, target_key: str) -> tuple[list[float], int]:
    values = []
    missing_boxed = 0
    for idx, row in enumerate(rows):
        if output_key not in row:
            raise SystemExit(f"Output key '{output_key}' is missing at row {idx}")
        if target_key not in row:
            raise SystemExit(f"Target key '{target_key}' is missing at row {idx}")

        pred = extract_boxed_answer(row[output_key])
        target = normalize_answer(row[target_key])
        if pred is None:
            missing_boxed += 1
            values.append(0.0)
            continue
        values.append(1.0 if answers_equal(pred, target) else 0.0)
    return values, missing_boxed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_PATH,
        help="Validation JSONL file, or a validation directory containing step JSONL files.",
    )
    parser.add_argument(
        "--mode",
        choices=("boxed", "metric"),
        default="boxed",
        help="Scoring mode. 'boxed' extracts the final answer from output \\boxed{} and compares with gts.",
    )
    parser.add_argument("--metric", choices=METRIC_PRIORITY, help="Metric to average. Defaults to acc, score, reward.")
    parser.add_argument("--output-key", default="output", help="JSONL field containing model output.")
    parser.add_argument("--target-key", default="gts", help="JSONL field containing ground truth.")
    args = parser.parse_args()

    path = resolve_input(args.path)
    rows = load_rows(path)

    missing_boxed = None
    if args.mode == "boxed":
        metric = "boxed_acc"
        values, missing_boxed = boxed_accuracy(rows, args.output_key, args.target_key)
    else:
        metric = pick_metric(rows, args.metric)
        values = numeric_values(rows, metric)

    score = mean(values)

    print(f"file: {path}")
    print(f"mode: {args.mode}")
    print(f"metric: {metric}")
    print(f"num_rows: {len(rows)}")
    print(f"num_metric_values: {len(values)}")
    if missing_boxed is not None:
        print(f"missing_boxed: {missing_boxed}")
    print(f"sum: {sum(values):.10g}")
    print(f"mean: {score:.10f}")
    print(f"percent: {100 * score:.4f}")


if __name__ == "__main__":
    main()
