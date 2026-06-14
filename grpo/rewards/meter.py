import json
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from rewards.common import (
    append_jsonl,
    clamp01,
    extract_text,
    logmean_prob,
    mean,
    load_env,
    poem_structure,
    pair_lines_to_bayts,
)


class BiLSTM4(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=4,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.emb(x)
        _, (h_n, _) = self.lstm(emb)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(h)


class BiLSTM4MeterPredictor:
    def __init__(self, model_id: str, hf_token: str = None, device: str = None):
        self.model_id = model_id
        self.hf_token = hf_token
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        bundle_path = hf_hub_download(repo_id=model_id, filename="bundle.pt", token=hf_token)
        bundle = torch.load(bundle_path, map_location="cpu")

        self.stoi = bundle["stoi"]
        self.classes = bundle["classes"]
        self.label2id = {lab: i for i, lab in enumerate(self.classes)}
        emb_dim = int(bundle.get("emb_dim", 32))
        hidden_dim = int(bundle.get("latent_dim", 64))
        num_classes = int(bundle.get("num_classes", len(self.classes)))
        self.T = int(bundle.get("max_len", bundle.get("max_seq_length", 128)))
        vocab_size = max(self.stoi.values()) + 1

        model = BiLSTM4(vocab_size, emb_dim, hidden_dim, num_classes, dropout=0.1)
        model.load_state_dict(bundle["model_state_dict"], strict=True)
        model.eval()
        self.model = model.to(self.device)

    def clean_to_stoi(self, text: str) -> str:
        text = str(text or "")
        text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("ـ", "")
        cleaned = []
        for ch in text:
            if ch in self.stoi:
                cleaned.append(ch)
            else:
                cleaned.append(" ")
        text = "".join(cleaned)
        return " ".join(text.split())

    def _encode(self, text: str) -> torch.Tensor:
        ids = np.zeros((self.T,), dtype=np.int64)
        for i, ch in enumerate(text[: self.T]):
            ids[i] = self.stoi.get(ch, 0)
        return torch.from_numpy(ids).unsqueeze(0)

    @torch.no_grad()
    def predict(self, text: str, target_meter: str) -> dict:
        x = self._encode(text).to(self.device)
        logits = self.model(x)[0]
        probs = torch.softmax(logits, dim=-1)
        pred_id = int(torch.argmax(probs).item())
        pred = self.classes[pred_id]
        target_id = self.label2id.get(target_meter)
        p_target = float(probs[target_id].item()) if target_id is not None else 0.0
        p_pred = float(probs[pred_id].item())
        return {
            "pred": pred,
            "p_target": p_target,
            "p_pred": p_pred,
        }


_PREDICTOR = None


def get_meter_predictor():
    global _PREDICTOR
    if _PREDICTOR is not None:
        return _PREDICTOR

    load_env()
    import os
    model_id = os.getenv("METER_MODEL_ID", "").strip()
    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    if not model_id:
        raise ValueError("METER_MODEL_ID is missing in .env")
    _PREDICTOR = BiLSTM4MeterPredictor(model_id=model_id, hf_token=hf_token)
    return _PREDICTOR


def resolve_target_meter(predictor, target_meter: str, base_meter: str = "") -> Dict[str, Any]:
    requested = str(target_meter or "").strip()
    base_meter = str(base_meter or "").strip()

    # Phase 1 policy: when a base meter is available, score against it directly.
    # This keeps the reward aligned with the classifier's label space.
    candidates = []
    if base_meter:
        candidates.append(("base_meter_only", base_meter))
    elif requested:
        candidates.append(("requested_meter_no_base", requested))
        if " " in requested:
            fallback = requested.split(" ", 1)[1].strip()
            if fallback and fallback != requested:
                candidates.append(("derived_from_requested_meter", fallback))

    for source, label in candidates:
        if label in predictor.label2id:
            return {
                "target_meter_requested": requested,
                "target_meter_used": label,
                "target_resolution": source,
                "candidate_labels": [lab for _, lab in candidates],
            }

    return {
        "target_meter_requested": requested,
        "target_meter_used": "",
        "target_resolution": "unresolved",
        "candidate_labels": [lab for _, lab in candidates],
    }


def score_meter_poem(generated_poem: str, target_meter: str, base_meter: str = "", aggregator: str = "logmean") -> Dict[str, Any]:
    predictor = get_meter_predictor()
    text = extract_text(generated_poem)
    struct = poem_structure(text)
    lines = struct["lines"]
    bayts = pair_lines_to_bayts(lines)
    target_info = resolve_target_meter(predictor, target_meter=target_meter, base_meter=base_meter)
    resolved_target = target_info["target_meter_used"]

    per_bayt = []
    per_bayt_details = []
    skipped = 0

    for bayt_index, bayt in enumerate(bayts):
        cleaned = predictor.clean_to_stoi(bayt)
        detail = {
            "bayt_index": bayt_index,
            "bayt_text": bayt,
            "cleaned_text": cleaned,
            "target_meter_requested": target_info["target_meter_requested"],
            "target_meter_used": resolved_target,
            "target_resolution": target_info["target_resolution"],
        }
        if len(cleaned.replace(" ", "")) < 20:
            skipped += 1
            detail["skipped"] = True
            detail["skip_reason"] = "too_short_after_cleaning"
            detail["p_target"] = None
            detail["p_pred"] = None
            detail["pred"] = ""
            per_bayt_details.append(detail)
            continue
        if not resolved_target:
            skipped += 1
            detail["skipped"] = True
            detail["skip_reason"] = "unresolved_target_meter"
            detail["p_target"] = 0.0
            detail["p_pred"] = None
            detail["pred"] = ""
            per_bayt_details.append(detail)
            continue
        pred = predictor.predict(cleaned, resolved_target)
        per_bayt.append(pred["p_target"])
        detail["skipped"] = False
        detail["skip_reason"] = ""
        detail.update(pred)
        per_bayt_details.append(detail)

    if aggregator == "mean":
        score = mean(per_bayt)
    else:
        score = logmean_prob(per_bayt)

    return {
        "score": clamp01(score),
        "requested_meter": str(target_meter or "").strip(),
        "base_meter": str(base_meter or "").strip(),
        "target_meter_used": resolved_target,
        "target_resolution": target_info["target_resolution"],
        "candidate_labels": target_info["candidate_labels"],
        "per_bayt_scores": per_bayt,
        "per_bayt_details": per_bayt_details,
        "num_valid_bayts": len(per_bayt),
        "num_skipped_bayts": skipped,
        "num_lines": struct["num_lines"],
        "complete_bayts": struct["complete_bayts"],
        "has_odd_tail": struct["has_odd_tail"],
        "odd_tail_line": struct["odd_tail_line"],
        "mean_score": mean(per_bayt),
        "logmean_score": logmean_prob(per_bayt),
        "std_score": float(np.std(per_bayt)) if per_bayt else 0.0,
        "min_score": float(min(per_bayt)) if per_bayt else 0.0,
        "max_score": float(max(per_bayt)) if per_bayt else 0.0,
    }


def make_meter_reward(run_dir: str, aggregator: str = "logmean", recorder=None):
    debug_path = f"{run_dir}/reward_meter_debug.jsonl"

    def reward_fn(prompts=None, completions=None, meter_label=None, base_meter=None, **kwargs):
        completions = completions or []
        meter_label = meter_label or kwargs.get("meter_label", [])
        base_meter = base_meter or kwargs.get("base_meter", [])
        if not isinstance(meter_label, list):
            meter_label = [meter_label] * len(completions)
        if not isinstance(base_meter, list):
            base_meter = [base_meter] * len(completions)

        rewards = []
        extras = []
        for i, completion in enumerate(completions):
            text = extract_text(completion)
            target = meter_label[i] if i < len(meter_label) else ""
            base = base_meter[i] if i < len(base_meter) else ""
            out = score_meter_poem(text, target, base_meter=base, aggregator=aggregator)
            rewards.append(out["score"])
            extra = {
                "i": i,
                "target_meter_requested": target,
                "base_meter": base,
                "target_meter_used": out["target_meter_used"],
                "target_resolution": out["target_resolution"],
                "score": out["score"],
                "mean_score": out["mean_score"],
                "logmean_score": out["logmean_score"],
                "std_score": out["std_score"],
                "min_score": out["min_score"],
                "max_score": out["max_score"],
                "num_valid_bayts": out["num_valid_bayts"],
                "num_skipped_bayts": out["num_skipped_bayts"],
                "num_lines": out["num_lines"],
                "complete_bayts": out["complete_bayts"],
                "has_odd_tail": out["has_odd_tail"],
                "odd_tail_line": out["odd_tail_line"],
                "candidate_labels": out["candidate_labels"],
                "per_bayt_details": out["per_bayt_details"],
                "text_preview": text[:400],
            }
            extras.append(extra)
            append_jsonl(debug_path, {"score": out["score"], **extra})
        if recorder is not None:
            recorder.record_reward_batch(
                reward_name="meter",
                completions=completions,
                prompts=prompts,
                scores=rewards,
                extras=extras,
                kwargs={
                    **kwargs,
                    "meter_label": meter_label,
                    "base_meter": base_meter,
                },
            )
        return rewards

    reward_fn.__name__ = "meter"
    return reward_fn
