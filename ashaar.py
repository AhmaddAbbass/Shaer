import json
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from safetensors.torch import load_file as load_safetensors
except Exception:  # pragma: no cover
    load_safetensors = None

from huggingface_hub import hf_hub_download


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, ff_dim, dropout):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
        )

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.ln1(x + self.drop(attn_out))
        ff_out = self.ffn(x)
        x = self.ln2(x + self.drop(ff_out))
        return x


class BiGRUResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.gru = nn.GRU(dim, dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.ln = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.ln(x + self.drop(out))


class MeterModel(nn.Module):
    def __init__(self, vocab_size, num_classes, T, emb_dim, num_heads, ff_dim, gru_blocks, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos = nn.Embedding(T, emb_dim)
        self.drop = nn.Dropout(dropout)
        self.tr = TransformerBlock(emb_dim, num_heads, ff_dim, dropout)
        self.gru_blocks = nn.ModuleList([BiGRUResidualBlock(emb_dim, dropout) for _ in range(gru_blocks)])
        self.head = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        bsz, t = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(bsz, t)
        h = self.drop(self.emb(x) + self.pos(pos))
        pad_mask = x == 0
        h = self.tr(h, key_padding_mask=pad_mask)
        for blk in self.gru_blocks:
            h = blk(h)
        mask = (~pad_mask).float().unsqueeze(-1)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return self.head(pooled)


class AshaarMeterPredictor:
    def __init__(self, model_id: str, hf_token: str | None = None, device: str | None = None):
        self.model_id = model_id
        self.hf_token = hf_token
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        config_path = hf_hub_download(repo_id=model_id, filename="config.json", token=hf_token)
        stoi_path = hf_hub_download(repo_id=model_id, filename="stoi.json", token=hf_token)

        weights_path = None
        try:
            weights_path = hf_hub_download(repo_id=model_id, filename="model.safetensors", token=hf_token)
        except Exception:
            weights_path = hf_hub_download(repo_id=model_id, filename="best_state_dict.pt", token=hf_token)

        config = json.load(open(config_path, "r", encoding="utf-8"))
        stoi = json.load(open(stoi_path, "r", encoding="utf-8"))
        classes = config.get("classes")
        if classes is None:
            label2id_path = hf_hub_download(repo_id=model_id, filename="label2id.json", token=hf_token)
            label2id = json.load(open(label2id_path, "r", encoding="utf-8"))
            classes = [None] * len(label2id)
            for k, v in label2id.items():
                classes[int(v)] = k

        self.stoi = stoi
        self.classes = classes
        self.label2id = {lab: i for i, lab in enumerate(classes)}

        self.T = int(config.get("max_length", config.get("max_len", 128)))
        self.vocab_size = int(config.get("vocab_size", max(stoi.values()) + 1))
        self.num_classes = int(config.get("num_classes", len(classes)))

        emb_dim = int(config.get("emb_dim", 64))
        num_heads = int(config.get("num_heads", 4))
        ff_dim = int(config.get("ff_dim", 256))
        gru_blocks = int(config.get("gru_blocks", 3))
        dropout = float(config.get("dropout", 0.1))

        model = MeterModel(self.vocab_size, self.num_classes, self.T, emb_dim, num_heads, ff_dim, gru_blocks, dropout)
        model.eval()

        if weights_path.endswith(".safetensors") and load_safetensors is not None:
            sd = load_safetensors(weights_path)
        else:
            sd = torch.load(weights_path, map_location="cpu")

        try:
            model.load_state_dict(sd, strict=True)
        except RuntimeError:
            sd2 = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
            model.load_state_dict(sd2, strict=True)

        self.model = model.to(self.device)

    def _encode(self, text: str) -> torch.Tensor:
        ids = np.zeros((self.T,), dtype=np.int64)
        for i, ch in enumerate(text[: self.T]):
            ids[i] = self.stoi.get(ch, 0)
        return torch.from_numpy(ids).unsqueeze(0)

    @torch.no_grad()
    def predict(self, text: str, target_meter: str, tau: float) -> Dict:
        x = self._encode(text).to(self.device)
        logits = self.model(x)[0]
        probs = torch.softmax(logits, dim=-1)
        pred_id = int(torch.argmax(probs).item())
        pred = self.classes[pred_id]
        target_id = self.label2id.get(target_meter)
        p_target = float(probs[target_id].item()) if target_id is not None else 0.0
        p_pred = p_target = float(probs[pred_id].item()) if target_id is not None else 0.0
        passed = int(pred == target_meter and p_target >= tau)
        return {
            "pred": pred,
            "p_target": p_target,
            "pass": passed,
            "p_pred":p_pred
        }
