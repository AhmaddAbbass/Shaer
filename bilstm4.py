import torch
import torch.nn as nn
import numpy as np
from huggingface_hub import hf_hub_download


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
    def __init__(self, model_id: str, hf_token: str | None = None, device: str | None = None):
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

    def _encode(self, text: str) -> torch.Tensor:
        ids = np.zeros((self.T,), dtype=np.int64)
        for i, ch in enumerate(text[: self.T]):
            ids[i] = self.stoi.get(ch, 0)
        return torch.from_numpy(ids).unsqueeze(0)

    @torch.no_grad()
    def predict(self, text: str, target_meter: str, tau: float) -> dict:
        x = self._encode(text).to(self.device)
        logits = self.model(x)[0]
        probs = torch.softmax(logits, dim=-1)
        pred_id = int(torch.argmax(probs).item())
        pred = self.classes[pred_id]
        target_id = self.label2id.get(target_meter)
        p_target = float(probs[target_id].item()) if target_id is not None else 0.0
        passed = int(pred == target_meter and p_target >= tau)
        return {
            "pred": pred,
            "p_target": p_target,
            "pass": passed,
        }
