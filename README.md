# 基于 CoLaKG 的 LastFM 音乐艺术家推荐系统

本项目基于 SIGIR 2025 论文 **Cui et al., Comprehending Knowledge Graphs with Large Language Models for Recommender Systems** 提出的 CoLaKG 框架，在 LastFM 数据集上完成表示学习训练，并构建了一个可直接运行的音乐艺术家推荐系统原型。项目核心思想是利用大语言模型（LLM）对知识图谱进行语义理解，学习高质量的用户与艺术家表示，并将这些表示迁移至下游的向量检索式推荐任务中。

本仓库完整复用了 CoLaKG 的表示学习结果，并实现了多种基于 embedding 的推荐策略、离线 Top-K 推荐评测以及基于 Streamlit 的交互式前端推荐系统。

---

## 1. 环境依赖（推荐方式）

本项目已导出完整的 Conda 环境文件，推荐直接使用：

```bash
conda env create -f environment.yml  
conda activate colakg 
```

说明：`environment.yml` 由实际运行环境导出，已包含 PyTorch、Streamlit、scikit-learn、transformers 等依赖，适用于 Linux（服务器或本地）。

---

## 2. 最快速启动方式（直接可跑）

本仓库已包含所需数据与 embedding 文件，clone 后可直接运行：

```
streamlit run app.py 
```

启动后终端会提示访问地址，例如：  http://localhost:8501  

### 远程服务器运行（SSH 端口转发）

```
ssh -L 8501:localhost:8501 your_user@server_ip 
```

然后在本地浏览器访问：  http://localhost:8501  

---

## 3. 数据集来源与数据文件说明

### 数据集来源

LastFM 数据集与数据组织方式均来自 CoLaKG 官方仓库：  https://github.com/ziqiangcui/CoLaKG-SIGIR25  

### data/ 目录结构

```text
data/  
└── lastfm/  
    ├── train.txt  
    ├── test.txt  
    ├── lastfm_embeddings_simcse_kg.pt  
    ├── lastfm_embeddings_simcse_kg_user.pt  
    └── colakg_trained_embeddings.pt  
```

文件说明：  

- `train.txt / test.txt`：用户–艺术家交互序列  
- `lastfm_embeddings_simcse_kg.pt`：LLM + KG 生成的艺术家语义 embedding  
- `astfm_embeddings_simcse_kg_user.pt`：用户语义 embedding  
- `colakg_trained_embeddings.pt`：CoLaKG 第二阶段训练后的最终 embedding（推荐使用）  

---

## 4. 前端系统功能说明

系统基于同一套 CoLaKG 训练后的 embedding，支持三种推荐模式：

1. 基于标签的推荐（Tag-to-Artist）：根据音乐风格标签返回代表性艺术家  
2. 基于相似艺术家的推荐（Artist-to-Artist）：输入若干喜欢的艺术家，返回风格相近的艺术家  
3. 基于历史偏好的推荐（History-aware Recommendation）：根据用户历史听歌顺序进行偏好聚合并推荐  

---

## 5. 训练 CoLaKG 模型（可选）

```
cd rec_code  
nohup bash train_lastfm_colakg.sh > logs/train_lastfm_colakg.log 2>&1 &  
```

模型 checkpoint 位于：  `code/checkpoints/ `

---

## 6. 导出训练后的 embedding（可选）

```
cd rec_code  
CUDA_VISIBLE_DEVICES=0 python export_colakg_emb.py --model colakg --dataset lastfm  
```

生成：  `data/lastfm/colakg_trained_embeddings.pt `

---

## 7. 离线评测

```
cd rec_code  
python eval_ours.py  
```

输出指标：  

- Recall@10 / Recall@20  
- NDCG@10 / NDCG@20  

---

## 8. 引用

> @inproceedings{colakg,  
>   title={Comprehending Knowledge Graphs with Large Language Models for Recommender Systems},  
>   author={Cui, Ziqiang and Weng, Yunpeng and Tang, Xing and others},  
>   booktitle={Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},  
>   year={2025}  
> }