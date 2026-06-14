yehimport re
import numpy as np
import random
from collections import Counter, defaultdict

def clean_to_stoi(text, stoi_keys):
    # normalize whitespace/newlines
    text = text.replace("\n", " ").replace("\t", " ").replace("\r", " ")
    text = text.replace("ـ", "")  # tatweel
    # keep only known characters, rest -> space
    text = "".join(ch if ch in stoi_keys else " " for ch in text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def hemistichs_to_bayts(verses):
    # pairs: (0,1), (2,3), ...
    bayts = []
    for i in range(0, len(verses) - 1, 2):
        bayts.append(verses[i] + " " + verses[i+1])
    return bayts

def softmax_np(x):
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-12)



import torch

@torch.no_grad()
def probs_for_text(predictor, text):
    x = predictor._encode(text).to(predictor.device)   # [1, T]
    logits = predictor.model(x)[0].detach().cpu().numpy()  # [C]
    return softmax_np(logits)  # [C]

def predict_poem_by_bayts(predictor, verses, agg="logmean"):
    stoi_keys = set(predictor.stoi.keys())

    bayts = hemistichs_to_bayts(verses)
    if len(bayts) == 0:
        return None  # too short / bad formatting

    probs_list = []
    preds = []

    for b in bayts:
        b = clean_to_stoi(b, stoi_keys)
        if len(b.replace(" ", "")) < 20:
            continue

        p = probs_for_text(predictor, b)
        probs_list.append(p)
        preds.append(predictor.classes[int(np.argmax(p))])

    if len(probs_list) == 0:
        return None

    P = np.stack(probs_list, axis=0)  # [num_bayts, C]

    if agg == "mean":
        p_agg = P.mean(axis=0)

    elif agg == "logmean":
        # average log-probabilities then exp -> proportional to product of evidence
        eps = 1e-12
        logp = np.log(P + eps).mean(axis=0)
        p_agg = np.exp(logp)
        p_agg = p_agg / (p_agg.sum() + 1e-12)

    elif agg == "vote":
        c = Counter(preds)
        top = c.most_common(1)[0][0]
        p_agg = np.zeros(P.shape[1], dtype=float)
        p_agg[predictor.label2id[top]] = 1.0

    else:
        raise ValueError("agg must be one of: mean, logmean, vote")

    pred = predictor.classes[int(np.argmax(p_agg))]
    return {
        "pred": pred,
        "p_vec": p_agg,
        "n_bayts_used": len(probs_list),
    }


import torch

@torch.no_grad()
def probs_for_text(predictor, text):
    x = predictor._encode(text).to(predictor.device)   # [1, T]
    logits = predictor.model(x)[0].detach().cpu().numpy()  # [C]
    return softmax_np(logits)  # [C]

def predict_poem_by_bayts(predictor, verses, agg="logmean"):
    stoi_keys = set(predictor.stoi.keys())

    bayts = hemistichs_to_bayts(verses)
    if len(bayts) == 0:
        return None  # too short / bad formatting

    probs_list = []
    preds = []

    for b in bayts:
        b = clean_to_stoi(b, stoi_keys)
        if len(b.replace(" ", "")) < 20:
            continue

        p = probs_for_text(predictor, b)
        probs_list.append(p)
        preds.append(predictor.classes[int(np.argmax(p))])

    if len(probs_list) == 0:
        return None

    P = np.stack(probs_list, axis=0)  # [num_bayts, C]

    if agg == "mean":
        p_agg = P.mean(axis=0)

    elif agg == "logmean":
        # average log-probabilities then exp -> proportional to product of evidence
        eps = 1e-12
        logp = np.log(P + eps).mean(axis=0)
        p_agg = np.exp(logp)
        p_agg = p_agg / (p_agg.sum() + 1e-12)

    elif agg == "vote":
        c = Counter(preds)
        top = c.most_common(1)[0][0]
        p_agg = np.zeros(P.shape[1], dtype=float)
        p_agg[predictor.label2id[top]] = 1.0

    else:
        raise ValueError("agg must be one of: mean, logmean, vote")

    pred = predictor.classes[int(np.argmax(p_agg))]
    return {
        "pred": pred,
        "p_vec": p_agg,
        "n_bayts_used": len(probs_list),
    }
import matplotlib.pyplot as plt

def evaluate_on_dataset(ds_split, predictor, n_per_meter=200, agg="logmean"):
    # group indices by meter
    idxs_by_meter = defaultdict(list)
    for i, m in enumerate(ds_split["base_meter"]):
        idxs_by_meter[m].append(i)

    # sample up to n_per_meter per meter
    sampled_idxs = []
    for m, idxs in idxs_by_meter.items():
        k = min(n_per_meter, len(idxs))
        sampled_idxs.extend(random.sample(idxs, k))

    scores_by_meter = defaultdict(list)
    acc_by_meter = defaultdict(list)

    for i in sampled_idxs:
        ex = ds_split[i]
        target = ex["base_meter"]
        verses = ex["poem verses"]

        out = predict_poem_by_bayts(predictor, verses, agg=agg)
        if out is None:
            continue

        pred = out["pred"]
        p_t = float(out["p_vec"][predictor.label2id[target]])

        scores_by_meter[target].append(p_t)
        acc_by_meter[target].append(1 if pred == target else 0)

    return scores_by_meter, acc_by_meter
def show_summary(scores_by_meter, acc_by_meter, title):
    meters = sorted(scores_by_meter.keys(), key=lambda m: -np.mean(scores_by_meter[m]))

    print("\n==", title, "==")
    for m in meters:
        mean_p = np.mean(scores_by_meter[m])
        acc = np.mean(acc_by_meter[m]) if len(acc_by_meter[m]) else 0.0
        n = len(scores_by_meter[m])
        print(f"{m:>10} | n={n:4d} | mean p(target)={mean_p:.3f} | acc={acc:.3f}")

    # hist per meter
    for m in meters:
        plt.figure(figsize=(6,3))
        plt.hist(scores_by_meter[m], bins=20)
        plt.title(f"{title}: p(target) — {m} (n={len(scores_by_meter[m])})")
        plt.xlabel("p(target)")
        plt.ylabel("count")
        plt.tight_layout()
        plt.show()
# Example:
scores_a, acc_a = evaluate_on_dataset(dataset["validation"], ashaar, n_per_meter=2000, agg="logmean")
show_summary(scores_a, acc_a, "Ashaar classifier (bayt-paired, logmean)")

scores_b, acc_b = evaluate_on_dataset(dataset["validation"], bilstm, n_per_meter=2000, agg="logmean")
show_summary(scores_b, acc_b, "4BiLSTM classifier (bayt-paired, logmean)")
# Archived Note

This note is still useful background on meter-classifier behavior. For current public usage in this repo, prefer:

- [evaluation/metrics.py](./evaluation/metrics.py)
- [grpo/rewards/meter.py](./grpo/rewards/meter.py)
