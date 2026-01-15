import torch
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

import world
import utils
import register
from register import dataset  # 复用训练入口创建的 dataset

# ===================== 训练同款配置（按你给的脚本） =====================
DATA_DIR = Path("../data/lastfm")

ITEM_SEM_PATH = DATA_DIR / "lastfm_embeddings_simcse_kg.pt"
USER_SEM_PATH = DATA_DIR / "lastfm_embeddings_simcse_kg_user.pt"

CKPT_PATH = Path("../code/checkpoints/colakg-lastfm-3-64.pth.tar")  # 你的实际ckpt
OUT_PATH  = DATA_DIR / "colakg_trained_embeddings.pt"

NEIGHBOR_K = 10  # 你训练脚本 neighbor_k=10
# =======================================================================

utils.set_seed(world.seed)
device = world.device
print(">>SEED:", world.seed)
print(">>device:", device)
print(">>model:", world.model_name, "dataset:", world.dataset)
print(">>ckpt:", CKPT_PATH.resolve())
print(">>semantic:", ITEM_SEM_PATH.resolve())

# 1) 加载语义 embedding（固定路径，不依赖 world.item_semantic_emb_file）
item_semantic_emb = torch.load(ITEM_SEM_PATH, map_location="cpu")
user_semantic_emb = torch.load(USER_SEM_PATH, map_location="cpu")

# 2) 构造语义近邻 sorted_indices（与 main.py 一致：cosine + topK，不含自身）
cosine_sim_matrix = cosine_similarity(item_semantic_emb.numpy())
sorted_indices = np.argsort(-cosine_sim_matrix, axis=1)
sorted_indices = sorted_indices[:, 1:NEIGHBOR_K + 1]
sorted_indices = torch.tensor(sorted_indices).long()

# 3) 构造模型（完全复用训练逻辑）
Recmodel = register.MODELS[world.model_name](
    world.config,
    dataset,
    sorted_indices,
    item_semantic_emb,
    user_semantic_emb,
).to(device)

# 4) 加载 checkpoint（兼容 .pth.tar 各种包装）
ckpt = torch.load(CKPT_PATH, map_location=device)

state = ckpt
if isinstance(ckpt, dict):
    for key in ["state_dict", "model", "net", "Recmodel", "model_state_dict"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            break

# 兼容 module. 前缀
if isinstance(state, dict):
    state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }

missing, unexpected = Recmodel.load_state_dict(state, strict=False)
print(">>load_state_dict done.")
print(">>missing keys:", len(missing))
print(">>unexpected keys:", len(unexpected))

# 5) 导出 embeddings（computer() 输出就是评测用最终表示）
Recmodel.eval()
with torch.no_grad():
    users_emb, items_emb = Recmodel.computer()

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {"user_emb": users_emb.detach().cpu(), "item_emb": items_emb.detach().cpu()},
    OUT_PATH,
)

print("✅ Export done:", OUT_PATH.resolve())
print("user_emb:", tuple(users_emb.shape), "item_emb:", tuple(items_emb.shape))
