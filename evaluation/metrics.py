#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRPO_ROOT = PROJECT_ROOT / "grpo"
if str(GRPO_ROOT) not in sys.path:
    sys.path.insert(0, str(GRPO_ROOT))


def _load_env_if_present() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv("/root/.env", override=False)
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _meter_label(base_meter: str, form: str, meter_label: str = "") -> str:
    meter_label = str(meter_label or "").strip()
    if meter_label:
        return meter_label
    base_meter = str(base_meter or "").strip()
    form = str(form or "").strip()
    if not form or form == "تام":
        return base_meter
    return f"{form} {base_meter}".strip()


def _requested_bayts(row: dict[str, Any]) -> int:
    for key in ("requested_bayts", "count_requested_bayts"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return max(0, int(value))
    lines = _requested_lines(row)
    return max(0, lines // 2)


def _requested_lines(row: dict[str, Any]) -> int:
    for key in ("requested_num_lines", "sft_num_lines", "requested_lines"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return max(0, int(value))
    bayts = row.get("requested_bayts")
    if bayts is not None and str(bayts).strip():
        return max(0, int(bayts) * 2)
    return 0


def score_generation_row(row: dict[str, Any]) -> dict[str, Any]:
    """Score one generated row using the existing GRPO 4BiLSTM meter path."""
    _load_env_if_present()
    from rewards.common import extract_text, score_count_adherence
    from rewards.meter import score_meter_poem

    generated_text = extract_text(row.get("generated_text") or row.get("completion") or "")
    requested_bayts = _requested_bayts(row)
    requested_lines = _requested_lines(row) or requested_bayts * 2
    base_meter = str(row.get("base_meter") or "").strip()
    form = str(row.get("form") or "").strip()
    meter_label = _meter_label(base_meter, form, str(row.get("meter_label") or ""))

    count = score_count_adherence(requested_bayts, generated_text)
    out: dict[str, Any] = {
        "count_adherence": float(count["score"]),
        "parsed_num_lines": int(count.get("num_lines", 0)),
        "requested_num_lines": int(requested_lines),
        "count_eval_status": "ok",
    }

    try:
        meter = score_meter_poem(generated_text, meter_label, base_meter=base_meter, aggregator="logmean")
        out.update(
            {
                "meter": float(meter["score"]),
                "meter_eval_status": "ok",
                "meter_predicted": _first_prediction(meter.get("per_bayt_details", [])),
                "meter_target_used": str(meter.get("target_meter_used", "")),
                "meter_confidence": float(meter.get("mean_score", 0.0)),
                "meter_num_valid_bayts": int(meter.get("num_valid_bayts", 0)),
                "meter_num_skipped_bayts": int(meter.get("num_skipped_bayts", 0)),
                "meter_details_json": json.dumps(
                    {
                        "target_resolution": meter.get("target_resolution", ""),
                        "per_bayt_scores": meter.get("per_bayt_scores", []),
                        "per_bayt_details": meter.get("per_bayt_details", []),
                        "has_odd_tail": meter.get("has_odd_tail", False),
                    },
                    ensure_ascii=False,
                ),
            }
        )
    except Exception as exc:
        out.update(
            {
                "meter": 0.0,
                "meter_eval_status": f"error:{type(exc).__name__}",
                "meter_predicted": "",
                "meter_target_used": "",
                "meter_confidence": 0.0,
                "meter_num_valid_bayts": 0,
                "meter_num_skipped_bayts": 0,
                "meter_details_json": json.dumps({"error": str(exc)}, ensure_ascii=False),
            }
        )
    return out


def _first_prediction(details: list[dict[str, Any]]) -> str:
    for detail in details or []:
        pred = str(detail.get("pred") or "").strip()
        if pred:
            return pred
    return ""
