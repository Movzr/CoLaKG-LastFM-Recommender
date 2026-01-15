#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_ours.py  (put in rec_code/)

Offline Top-K evaluation for LastFM using provided train/test split:
- Each line: user_id item1 item2 ...
- Evaluate two simple user-profile methods:
  1) Ours-Mean: user vector = mean of train item embeddings
  2) Ours-Weighted: user vector = weighted sum (earlier items get larger weight)

Embedding source (NO args needed):
- Default: CoLaKG trained item embeddings exported to:
    ../data/lastfm/colakg_trained_embeddings.pt  (dict with key "item_emb")
- Fallback: semantic embeddings:
    ../data/lastfm/lastfm_embeddings_simcse_kg.pt

Metrics:
- Recall@10/20
- NDCG@10/20

Run:
  cd rec_code
  python eval_ours.py
"""

import math
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch


# ===================== Fixed paths (no args) =====================
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "lastfm"
TRAIN_PATH = DATA_DIR / "train.txt"
TEST_PATH = DATA_DIR / "test.txt"

# Prefer CoLaKG trained embeddings (export_colakg_emb.py output)
COLAKG_EMB_PATH = DATA_DIR / "colakg_trained_embeddings.pt"  # dict: {"user_emb","item_emb"}
# Fallback to semantic embeddings
SEMANTIC_EMB_PATH = DATA_DIR / "lastfm_embeddings_simcse_kg.pt"

TOPKS = [10, 20]
BATCH_USERS = 256
DEVICE = "cuda"
# ================================================================


def read_user_item_list(file_path: Path) -> Dict[int, List[int]]:
    """
    Read format:
      u i1 i2 i3 ...
    Returns dict[u] = [i1, i2, ...]
    """
    data: Dict[int, List[int]] = {}
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            arr = list(map(int, line.split()))
            if len(arr) <= 1:
                continue
            u = arr[0]
            items = arr[1:]
            data[u] = items
    return data


def load_item_embeddings_auto() -> torch.Tensor:
    """
    Prefer CoLaKG trained embeddings if present, else use semantic embeddings.
    Returns: [num_items, dim] float32 tensor on CPU.
    """
    if COLAKG_EMB_PATH.exists():
        obj = torch.load(COLAKG_EMB_PATH, map_location="cpu")
        if not isinstance(obj, dict) or "item_emb" not in obj:
            raise ValueError(
                f"{COLAKG_EMB_PATH} exists but is not a dict with key 'item_emb'. "
                f"Got type={type(obj)}, keys={list(obj.keys()) if isinstance(obj, dict) else None}"
            )
        emb = obj["item_emb"]
        if not isinstance(emb, torch.Tensor):
            raise TypeError(f"obj['item_emb'] is not a Tensor, got {type(emb)}")
        print(f"[INFO] Using CoLaKG trained item_emb: {COLAKG_EMB_PATH}")
    else:
        obj = torch.load(SEMANTIC_EMB_PATH, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            emb = obj
        elif isinstance(obj, dict):
            # try common keys for semantic embedding files
            for k in ["item_emb", "item_embedding", "item_embeddings", "emb", "embeddings", "items", "item"]:
                if k in obj and isinstance(obj[k], torch.Tensor):
                    emb = obj[k]
                    break
            else:
                raise ValueError(f"Cannot find tensor in {SEMANTIC_EMB_PATH}, keys={list(obj.keys())[:20]}")
        else:
            raise ValueError(f"Unsupported embedding object type: {type(obj)}")
        print(f"[INFO] Using semantic embeddings: {SEMANTIC_EMB_PATH}")

    emb = emb.float().contiguous()
    if emb.dim() != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {tuple(emb.shape)}")
    return emb


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def recall_at_k(pred: List[int], gt_set: Set[int], k: int) -> float:
    if not gt_set:
        return 0.0
    hit = 0
    for i in pred[:k]:
        if i in gt_set:
            hit += 1
    return hit / float(len(gt_set))


def ndcg_at_k(pred: List[int], gt_set: Set[int], k: int) -> float:
    if not gt_set:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(pred[:k], start=1):
        if item in gt_set:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_len = min(len(gt_set), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_len + 1))
    return dcg / idcg if idcg > 0 else 0.0


def build_user_vectors_mean(
    train_dict: Dict[int, List[int]],
    item_emb: torch.Tensor
) -> Tuple[List[int], torch.Tensor]:
    users = []
    vecs = []
    for u, items in train_dict.items():
        if len(items) == 0:
            continue
        users.append(u)
        idx = torch.tensor(items, dtype=torch.long)
        v = item_emb.index_select(0, idx).mean(dim=0)
        vecs.append(v)
    return users, torch.stack(vecs, dim=0) if vecs else torch.empty((0, item_emb.size(1)))


def build_user_vectors_weighted_earlier_more(
    train_dict: Dict[int, List[int]],
    item_emb: torch.Tensor
) -> Tuple[List[int], torch.Tensor]:
    """
    weights: earlier items higher:
      raw = [n, n-1, ..., 1] then normalized to sum=1
    """
    users = []
    vecs = []
    for u, items in train_dict.items():
        n = len(items)
        if n == 0:
            continue
        users.append(u)
        idx = torch.tensor(items, dtype=torch.long)
        E = item_emb.index_select(0, idx)  # [n, d]
        raw = torch.arange(n, 0, -1, dtype=torch.float32)  # [n, ..., 1]
        w = raw / raw.sum()
        v = (E * w.unsqueeze(1)).sum(dim=0)
        vecs.append(v)
    return users, torch.stack(vecs, dim=0) if vecs else torch.empty((0, item_emb.size(1)))


def recommend_topk_all_items(
    user_vecs: torch.Tensor,
    item_emb_norm: torch.Tensor,
    train_items_per_user: List[Set[int]],
    topk_max: int,
) -> List[List[int]]:
    """
    user_vecs: [B, d] (already normalized)
    item_emb_norm: [M, d] normalized
    train_items_per_user: length B, set of items to exclude
    returns list of predicted item ids for each user, length topk_max
    """
    scores = user_vecs @ item_emb_norm.t()  # [B, M]
    # exclude train positives
    for i, seen in enumerate(train_items_per_user):
        if seen:
            seen_idx = torch.tensor(sorted(seen), dtype=torch.long, device=scores.device)
            scores[i, seen_idx] = -1e9
    _, top_idx = torch.topk(scores, k=topk_max, dim=1)
    return top_idx.detach().cpu().tolist()


def evaluate(
    train_dict: Dict[int, List[int]],
    test_dict: Dict[int, List[int]],
    item_emb: torch.Tensor,
    topks: List[int],
    method: str,
    batch_users: int,
    device: str,
) -> Dict[str, float]:
    assert method in ["mean", "weighted"], method
    topk_max = max(topks)

    item_emb_norm = l2_normalize(item_emb).to(device)

    if method == "mean":
        users, uvec = build_user_vectors_mean(train_dict, item_emb)
    else:
        users, uvec = build_user_vectors_weighted_earlier_more(train_dict, item_emb)

    eval_users = [u for u in users if u in test_dict and len(test_dict[u]) > 0 and len(train_dict.get(u, [])) > 0]
    if not eval_users:
        raise RuntimeError("No valid users to evaluate (check train/test format).")

    user_to_idx = {u: i for i, u in enumerate(users)}
    idxs = [user_to_idx[u] for u in eval_users]
    uvec = uvec.index_select(0, torch.tensor(idxs, dtype=torch.long))

    train_sets = [set(train_dict[u]) for u in eval_users]
    gt_sets = [set(test_dict[u]) for u in eval_users]

    uvec = l2_normalize(uvec).to(device)

    sum_recall = {k: 0.0 for k in topks}
    sum_ndcg = {k: 0.0 for k in topks}

    n_users = len(eval_users)
    for start in range(0, n_users, batch_users):
        end = min(start + batch_users, n_users)
        batch_uvec = uvec[start:end]
        batch_train_sets = train_sets[start:end]
        batch_gt_sets = gt_sets[start:end]

        preds = recommend_topk_all_items(
            user_vecs=batch_uvec,
            item_emb_norm=item_emb_norm,
            train_items_per_user=batch_train_sets,
            topk_max=topk_max,
        )

        for p, gt in zip(preds, batch_gt_sets):
            for k in topks:
                sum_recall[k] += recall_at_k(p, gt, k)
                sum_ndcg[k] += ndcg_at_k(p, gt, k)

    results = {}
    for k in topks:
        results[f"R@{k}"] = sum_recall[k] / n_users
        results[f"N@{k}"] = sum_ndcg[k] / n_users
    results["users"] = n_users
    return results


def main():
    device = DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing {TRAIN_PATH}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing {TEST_PATH}")

    train_dict = read_user_item_list(TRAIN_PATH)
    test_dict = read_user_item_list(TEST_PATH)

    item_emb = load_item_embeddings_auto()
    num_items, dim = item_emb.shape

    print(f"[INFO] Loaded item embeddings: shape={tuple(item_emb.shape)}")
    print(f"[INFO] Loaded train users={len(train_dict)}, test users={len(test_dict)}")
    print(f"[INFO] Eval topks={TOPKS}, batch_users={BATCH_USERS}, device={device}")

    # sanity check: item ids range
    max_train_item = max((max(v) for v in train_dict.values() if v), default=-1)
    max_test_item = max((max(v) for v in test_dict.values() if v), default=-1)
    max_item = max(max_train_item, max_test_item)
    if max_item >= num_items:
        raise ValueError(f"Item id out of range: max_item_id={max_item} but emb has num_items={num_items}")

    res_mean = evaluate(train_dict, test_dict, item_emb, TOPKS, method="mean",
                        batch_users=BATCH_USERS, device=device)
    res_w = evaluate(train_dict, test_dict, item_emb, TOPKS, method="weighted",
                     batch_users=BATCH_USERS, device=device)

    print("\n================= RESULTS (LastFM) =================")
    print(f"Ours-Mean      users={res_mean['users']}")
    for k in TOPKS:
        print(f"  R@{k}: {res_mean[f'R@{k}']:.4f}   N@{k}: {res_mean[f'N@{k}']:.4f}")

    print(f"\nOurs-Weighted  users={res_w['users']}")
    for k in TOPKS:
        print(f"  R@{k}: {res_w[f'R@{k}']:.4f}   N@{k}: {res_w[f'N@{k}']:.4f}")
    print("====================================================\n")


if __name__ == "__main__":
    main()
