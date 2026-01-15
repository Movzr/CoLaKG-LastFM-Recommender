# simple_lastfm_recommender.py
# 基于 CoLaKG 提供的 lastfm_embeddings_simcse_kg.pt 做一个简单推荐器

from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn.functional as F

from lastfm_utils import (
    load_lastfm_train,
    build_mapped_item_metadata,
)


# 路径：CoLaKG-SIGIR25/data/lastfm
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "lastfm"

SEMANTIC_EMB_PATH = DATA_DIR / "lastfm_embeddings_simcse_kg.pt"
COLAKG_EMB_PATH   = DATA_DIR / "colakg_trained_embeddings.pt"


def load_item_embeddings(source: str = "colakg") -> torch.Tensor:
    """
    source:
        - "semantic": LLM / SimCSE / KG 语义 embedding（原来的）
        - "colakg":   CoLaKG 训练后的 final item embedding
    """
    if source == "colakg":
        emb_path = COLAKG_EMB_PATH
    else:
        emb_path = SEMANTIC_EMB_PATH

    obj = torch.load(emb_path, map_location="cpu")

    # ===== 原有兼容逻辑（完全保留）=====
    if isinstance(obj, torch.Tensor):
        item_emb = obj
    elif isinstance(obj, dict):
        for key in ["item_emb", "item_embedding", "items", "emb", "embeddings"]:
            if key in obj:
                item_emb = obj[key]
                break
        else:
            raise ValueError(
                f"Unknown dict format in {emb_path}, keys = {list(obj.keys())}"
            )
    else:
        raise TypeError(f"Unsupported type from {emb_path}: {type(obj)}")

    if item_emb.ndim != 2:
        raise ValueError(f"item_emb should be 2D, got {item_emb.shape}")

    return item_emb



def recommend_for_liked_items(
    liked_items: List[int],
    item_emb: torch.Tensor,
    item_meta: Dict[int, Dict[str, Any]],
    topk: int = 20,
) -> List[Dict[str, Any]]:
    """
    输入:
        liked_items: 用户喜欢的若干 mapped_item_id
        item_emb: [num_items, dim]，与 mapped_item_id 对齐
        item_meta: mapped_item_id -> {name, url, pictureURL, raw_id}
        topk: 返回多少首推荐

    输出:
        一个列表，每个元素是:
        {
            "item_id": mapped_item_id,
            "score": float 相似度,
            "name": 歌手名,
            "url": LastFM 链接,
            "pictureURL": 封面图,
        }
    """
    if len(liked_items) == 0:
        raise ValueError("liked_items 不能为空")

    device = item_emb.device
    emb_norm = F.normalize(item_emb, dim=-1)   # [N, D]

    liked = torch.tensor(liked_items, dtype=torch.long, device=device)
    assert liked.max().item() < emb_norm.size(0), "liked item_id 超出 embedding 范围"

    liked_vecs = emb_norm[liked]                     # [L, D]
    user_vec = liked_vecs.mean(dim=0, keepdim=True)  # [1, D]

    # 所有 item 与 user_vec 的余弦相似度
    scores = torch.matmul(emb_norm, user_vec.t()).squeeze(-1)  # [N]

    # 把已喜欢的 item 排除掉
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask[liked] = False
    scores = scores.masked_fill(~mask, -1e9)

    topk = min(topk, scores.size(0))
    top_scores, top_idx = torch.topk(scores, k=topk, dim=0)

    results: List[Dict[str, Any]] = []
    for item_id, score in zip(top_idx.tolist(), top_scores.tolist()):
        meta = item_meta.get(item_id, {})
        results.append(
            {
                "item_id": item_id,
                "score": float(score),
                "name": meta.get("name", f"Item_{item_id}"),
                "url": meta.get("url", ""),
                "pictureURL": meta.get("pictureURL", ""),
                "raw_id": meta.get("raw_id", None),
            }
        )
    return results


def recommend_for_tags(
    tag_names: List[str],
    item_emb: torch.Tensor,
    item_meta: Dict[int, Dict[str, Any]],
    tagname_to_items: Dict[str, List[int]],
    topk: int = 20,
) -> List[Dict[str, Any]]:
    """
    根据若干 tag_name 推荐艺术家（简单版：按“被这些 tag 标记过的次数”排序）

    思路：
      - 对每个 tag，从 tagname_to_items 中取出该 tag 下的 mapped_item_id 列表
      - 合并所有列表，对每个 item 统计出现次数
      - 按出现次数排序，返回 Top-K

    这里暂时不用 embedding，只做一个“基于标签的直观推荐”，
    后面如果你想升级成“tag 语义向量”也很容易。
    """
    from collections import Counter

    all_items: List[int] = []
    for t in tag_names:
        items = tagname_to_items.get(t, [])
        all_items.extend(items)

    if not all_items:
        return []

    counter = Counter(all_items)  # item_id -> count

    ranked = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:topk]

    results: List[Dict[str, Any]] = []
    for item_id, cnt in ranked:
        meta = item_meta.get(item_id, {})
        results.append(
            {
                "item_id": item_id,
                "score": float(cnt),  # 用“被这些标签命中过的次数”当分数
                "name": meta.get("name", f"Item_{item_id}"),
                "url": meta.get("url", ""),
                "pictureURL": meta.get("pictureURL", ""),
                "raw_id": meta.get("raw_id", None),
            }
        )
    return results


def demo_use_train_user(user_id: int = 0, topk: int = 20):
    """
    用 train.txt 里的一个真实用户做演示：
      1) 读取该用户在 train 中交互过的 item
      2) 将这些 item 当作“已喜欢”
      3) 基于语义 embedding 做 Top-K 推荐
    """
    print("[DEMO] 使用 train.txt 中的用户 {} 做示例推荐".format(user_id))

    user2items = load_lastfm_train()
    if user_id not in user2items:
        raise ValueError("user_id {} 不在 train.txt 中".format(user_id))

    liked_items = user2items[user_id]
    print("该用户在 train 中交互过 {} 个 item，前 10 个: {}".format(
        len(liked_items), liked_items[:10])
    )

    item_meta = build_mapped_item_metadata()
    item_emb = load_item_embeddings()

    recs = recommend_for_liked_items(
        liked_items=liked_items,
        item_emb=item_emb,
        item_meta=item_meta,
        topk=topk,
    )

    print("\n为用户 {} 推荐 Top-{}：\n".format(user_id, topk))
    for i, r in enumerate(recs, start=1):
        print(
            "[{:02d}] score={:.4f} | id={} | name={} | url={}".format(
                i, r["score"], r["item_id"], r["name"], r["url"]
            )
        )

def recommend_for_liked_items_weighted(
    liked_items: List[int],
    weights: List[float],
    item_emb: torch.Tensor,
    item_meta: Dict[int, Dict[str, Any]],
    topk: int = 20,
) -> List[Dict[str, Any]]:
    """
    带权重的“按听歌历史推荐”：
      - liked_items: 按顺序排列的艺术家 ID 列表（历史序列）
      - weights: 与 liked_items 等长的权重列表（会自动归一化）
        （例如，越靠前的历史权重越高）
    """
    if len(liked_items) == 0:
        raise ValueError("liked_items 不能为空")
    if len(liked_items) != len(weights):
        raise ValueError("liked_items 与 weights 长度必须一致")

    device = item_emb.device
    emb_norm = F.normalize(item_emb, dim=-1)   # [N, D]

    idx_tensor = torch.tensor(liked_items, dtype=torch.long, device=device)
    assert idx_tensor.max().item() < emb_norm.size(0), "liked item_id 超出 embedding 范围"

    hist_vecs = emb_norm[idx_tensor]  # [L, D]

    # 归一化权重
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    if w.sum().item() <= 0:
        w = torch.ones_like(w)
    w = w / w.sum()  # 和为 1

    # 加权求和得到“用户历史表示”
    # shape: [D] -> [1,D]
    user_vec = (hist_vecs * w.unsqueeze(-1)).sum(dim=0, keepdim=True)  # [1, D]

    # 所有 item 与 user_vec 的余弦相似度（因为已经 normalize 了，这里可以直接点积）
    scores = torch.matmul(emb_norm, user_vec.t()).squeeze(-1)  # [N]

    # 把历史里出现过的 item 排除掉，防止推荐历史本身
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask[idx_tensor] = False
    scores = scores.masked_fill(~mask, -1e9)

    topk = min(topk, scores.size(0))
    top_scores, top_idx = torch.topk(scores, k=topk, dim=0)

    results: List[Dict[str, Any]] = []
    for item_id, score in zip(top_idx.tolist(), top_scores.tolist()):
        meta = item_meta.get(item_id, {})
        results.append(
            {
                "item_id": item_id,
                "score": float(score),
                "name": meta.get("name", f"Item_{item_id}"),
                "url": meta.get("url", ""),
                "pictureURL": meta.get("pictureURL", ""),
                "raw_id": meta.get("raw_id", None),
            }
        )
    return results


if __name__ == "__main__":
    demo_use_train_user(user_id=0, topk=20)
