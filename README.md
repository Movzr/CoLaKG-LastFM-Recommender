# 基于 CoLaKG 表示学习的 LastFM 音乐艺术家推荐系统

本项目基于 SIGIR 2025 提出的 **CoLaKG** 框架，在 LastFM 数据集上完成表示学习训练，并在导出用户/物品 embedding 后，构建了一个 **基于向量检索的多模式音乐艺术家推荐系统原型**。系统同时包含离线评测脚本与基于 Streamlit 的交互式前端，支持多种推荐方式。

---

## 1. 环境依赖

### Python 环境
建议使用 Python 3.8+ 并创建独立虚拟环境：

conda create -n colakg python=3.8 -y  
conda activate colakg  

### 依赖安装
根据项目实际使用情况，安装如下依赖（服务器请按 CUDA 版本安装 PyTorch）：

pip install -U pip  
pip install numpy scipy pandas scikit-learn tqdm  
pip install torch torchvision torchaudio  
pip install streamlit  
pip install tensorboardX  

---

## 2. 数据准备

项目默认使用 CoLaKG 官方仓库中 LastFM 的数据组织方式，目录结构示例如下：

CoLaKG-SIGIR25/  
├── data/  
│   └── lastfm/  
│       ├── train.txt  
│       ├── test.txt  
│       ├── lastfm_embeddings_simcse_kg.pt  
│       ├── lastfm_embeddings_simcse_kg_user.pt  
│       └── colakg_trained_embeddings.pt  

说明：
- train.txt / test.txt：每行格式为 `user_id item1 item2 ...`
- lastfm_embeddings_simcse_kg.pt：第一阶段由 LLM + KG 生成的语义 embedding
- colakg_trained_embeddings.pt：CoLaKG 第二阶段训练后导出的最终 embedding（**推荐使用**）

---

## 3. 训练 CoLaKG 模型（可选）

若需要完整复现实验流程，可在 `rec_code/` 目录下启动训练：

cd rec_code  
nohup bash train_lastfm_colakg.sh > logs/train_lastfm_colakg.log 2>&1 &  

训练完成后，模型 checkpoint 通常保存在：

code/checkpoints/colakg-lastfm-3-64.pth.tar  

---

## 4. 导出训练后的 embedding（推荐）

在 `rec_code/` 目录下运行：

cd rec_code  
CUDA_VISIBLE_DEVICES=0 python export_colakg_emb.py --model colakg --dataset lastfm  

成功后将在 `data/lastfm/` 目录下生成训练后的 embedding 文件，例如：

colakg_trained_embeddings.pt  

---

## 5. 离线评测（Offline Evaluation）

将评测脚本（如 eval_ours.py）放置于 `rec_code/` 目录，直接运行：

cd rec_code  
python eval_ours.py  

默认行为：
- 使用 train.txt / test.txt
- 读取 colakg_trained_embeddings.pt
- 输出 Recall@10、Recall@20、NDCG@10、NDCG@20

---

## 6. 启动前端系统（Streamlit）

前端入口文件为 app.py，位于 `rec_code/` 目录下，启动方式如下：

cd rec_code  
streamlit run app.py  

启动后终端会提示访问地址，例如：
http://localhost:8501  

### 服务器部署（SSH 端口转发）
若在远程服务器运行，可在本地执行：

ssh -L 8501:localhost:8501 your_user@server_ip  

随后在本地浏览器访问：
http://localhost:8501  

---

## 7. 前端功能说明

系统基于 **同一套 CoLaKG 训练后的 embedding 表示**，支持以下三种推荐模式：

1. **基于标签的推荐（Tag-to-Artist）**  
   用户选择音乐风格标签（如 rock、metal），系统返回语义相关的代表性艺术家。

2. **基于相似艺术家的推荐（Artist-to-Artist）**  
   用户输入若干已知喜好的艺术家，系统基于 embedding 相似度推荐风格相近的艺术家。

3. **基于历史偏好的推荐（History-aware Recommendation）**  
   用户按时间顺序输入历史听歌艺术家列表，系统通过偏好向量聚合并进行 Top-K 检索。

---

## 8. 常见问题

- **推荐使用哪种 embedding？**  
  建议使用 `colakg_trained_embeddings.pt`（CoLaKG 第二阶段训练后的最终表示）。

- **embedding 路径在哪里修改？**  
  通常在代码中类似如下位置修改：  
  DATA_DIR = data/lastfm  
  COLAKG_EMB_PATH = colakg_trained_embeddings.pt  

- **如何查找模型 checkpoint？**  
  可使用：  
  find . -name "*.pth" -o -name "*.pth.tar"  

---

## 9. 引用

如在课程报告或论文中引用 CoLaKG，请使用以下文献：

Cui et al., *Comprehending Knowledge Graphs with Large Language Models for Recommender Systems*, SIGIR 2025.
