# app.py  (放在 rec_code/)
# 三种模式：
#   1. 按标签推荐艺术家
#   2. 按艺术家推荐相似艺术家（等权重）
#   3. 按听歌历史推荐（顺序 + 加权：越早出现权重越大）

import streamlit as st

from lastfm_utils import (
    build_mapped_item_metadata,
    build_tag_to_mapped_items,
)
from simple_lastfm_recommender import (
    load_item_embeddings,
    recommend_for_liked_items,
    recommend_for_tags,
    recommend_for_liked_items_weighted,
)


@st.cache_resource
def get_data():
    """
    一次性把需要的东西都加载好：
      - item_meta:  item_id -> {name, url, pictureURL, raw_id}
      - item_emb:   [num_items, dim]
      - tagname_to_items: tag_name -> [item_id, ...]
    """
    item_meta = build_mapped_item_metadata()
    item_emb = load_item_embeddings("colakg")
    tagname_to_items = build_tag_to_mapped_items()
    return item_meta, item_emb, tagname_to_items


def build_name_index(item_meta):
    """
    根据 item_meta 构建：
      - id2name: item_id -> 艺术家名
      - name2id: 艺术家名 -> item_id（如果重名，只保留第一个）
    """
    id2name = {}
    name2id = {}
    for iid, meta in item_meta.items():
        name = meta.get("name", f"Item_{iid}")
        id2name[iid] = name
        if name not in name2id:
            name2id[name] = iid
    return id2name, name2id


def main():
    st.set_page_config(
        page_title="LastFM 音乐推荐 Demo",
        layout="wide"
    )

    st.title("🎵 LastFM 音乐推荐 Demo")

    st.markdown(
        """
本 Demo 基于 CoLaKG 提供的语义向量与 LastFM 标签数据，提供三种推荐方式：

1. **按标签推荐艺术家**：选择风格标签（如 `metal` / `rock` / `pop`），推荐典型代表  
2. **按艺术家推荐相似艺术家**：选择你喜欢的一批艺术家，推荐相似的艺术家  
3. **按听歌历史推荐**：按时间顺序输入听歌历史，**越早出现的艺术家权重越大**，模拟长期偏好  
"""
    )

    with st.spinner("加载数据中..."):
        item_meta, item_emb, tagname_to_items = get_data()
        id2name, name2id = build_name_index(item_meta)

    # ---------- 模式选择：三个按钮并排 ----------
    if "mode" not in st.session_state:
        st.session_state["mode"] = "tag"  # 默认按标签

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("按标签推荐"):
            st.session_state["mode"] = "tag"
    with col2:
        if st.button("按艺术家推荐"):
            st.session_state["mode"] = "artist"
    with col3:
        if st.button("按听歌历史推荐"):
            st.session_state["mode"] = "history"

    mode = st.session_state["mode"]

    # ===================== 模式一：按标签推荐 =====================
    if mode == "tag":
        st.subheader("🎯 模式一：按标签推荐艺术家")

        all_tags = sorted(tagname_to_items.keys())
        default_tags = all_tags[:5] if len(all_tags) >= 5 else all_tags

        selected_tags = st.multiselect(
            "请选择一个或多个标签（例如 metal, rock, pop...）：",
            options=all_tags,
            default=default_tags,
        )

        st.write(f"当前选择了 {len(selected_tags)} 个标签。")

        topk = st.slider("推荐 Top-K", 5, 50, 20, key="tag_topk")

        if st.button("生成标签推荐"):
            if not selected_tags:
                st.warning("请至少选择一个标签。")
            else:
                recs = recommend_for_tags(
                    tag_names=selected_tags,
                    item_emb=item_emb,
                    item_meta=item_meta,
                    tagname_to_items=tagname_to_items,
                    topk=topk,
                )
                if not recs:
                    st.warning("这些标签下没有找到对应的艺术家，换几个标签试试？")
                else:
                    st.success("推荐结果：")
                    for idx, r in enumerate(recs, 1):
                        st.markdown(
                            f"**[{idx}] {r['name']}**（被选中标签命中次数: {r['score']:.0f}）"
                        )
                        if r["pictureURL"]:
                            st.image(r["pictureURL"], width=120)
                        if r["url"]:
                            st.markdown(f"- 链接：[LastFM]({r['url']})")
                        st.write("---")

    # ===================== 模式二：按艺术家推荐（等权重） =====================
    elif mode == "artist":
        st.subheader("🎧 模式二：推荐相似艺术家")

        names = sorted(name2id.keys())
        default_names = names[:5] if len(names) >= 5 else names

        selected = st.multiselect(
            "请选择你喜欢的艺术家（顺序不区分，等权重）：",
            names,
            default=default_names,
        )
        liked_ids = [name2id[n] for n in selected]

        st.write(f"当前选择了 {len(liked_ids)} 个艺术家。")

        topk = st.slider("推荐 Top-K", 5, 50, 20, key="artist_topk")

        if st.button("生成相似艺术家推荐（等权重）"):
            if not liked_ids:
                st.warning("请至少选择一个艺术家！")
            else:
                recs = recommend_for_liked_items(
                    liked_items=liked_ids,
                    item_emb=item_emb,
                    item_meta=item_meta,
                    topk=topk,
                )
                st.success("推荐结果：")
                for idx, r in enumerate(recs, 1):
                    st.markdown(
                        f"**[{idx}] {r['name']}**（相似度: {r['score']:.4f}）"
                    )
                    if r["pictureURL"]:
                        st.image(r["pictureURL"], width=120)
                    if r["url"]:
                        st.markdown(f"- 链接：[LastFM]({r['url']})")
                    st.write("---")

    # ===================== 模式三：按听歌历史推荐（顺序 + 权重） =====================
    elif mode == "history":
        st.subheader("📜 模式三：按听歌历史推荐（顺序加权）")

        st.markdown(
            """
请在下面文本框中**按时间顺序**输入你的听歌历史，每一行一个艺术家名：

- **越早出现的艺术家权重越大**，模拟“长期偏好更重要”的场景  
- 示例：

Coldplay  
Radiohead  
Muse  

表示你最早听 Coldplay，后来依次听了 Radiohead、Muse。
"""
        )

        history_text = st.text_area(
            "输入听歌历史（每行一个艺术家名）：",
            value="",
            height=180,
            placeholder="例如：\nColdplay\nRadiohead\nMuse",
        )

        parsed_names = []
        if history_text.strip():
            for line in history_text.splitlines():
                name = line.strip()
                if name:
                    parsed_names.append(name)

        # 映射到 ID，并统计哪些没匹配上
        history_ids = []
        unknown_names = []
        for name in parsed_names:
            if name in name2id:
                history_ids.append(name2id[name])
            else:
                unknown_names.append(name)

        st.write(f"解析到 {len(parsed_names)} 个名字，其中 {len(history_ids)} 个在数据集中找到。")

        if unknown_names:
            st.warning(
                "以下艺术家名在 LastFM 数据集中未找到，将被忽略：\n"
                + ", ".join(unknown_names[:10])
                + (" ..." if len(unknown_names) > 10 else "")
            )

        # 生成权重：越早出现权重越大
        weights = []
        if history_ids:
            n = len(history_ids)
            raw_weights = [n - i for i in range(n)]  # [n, n-1, ..., 1]
            s = float(sum(raw_weights))
            weights = [w / s for w in raw_weights]

            st.write("为历史中每个艺术家分配的相对权重（和=1）：")
            for name, w in zip(parsed_names, weights):
                st.write(f"- {name}: {w:.3f}")

        topk = st.slider("推荐 Top-K", 5, 50, 20, key="history_topk")

        if st.button("生成按听歌历史的推荐（顺序加权）"):
            if not history_ids:
                st.warning("请至少输入一个能在数据集中匹配到的艺术家名字。")
            else:
                recs = recommend_for_liked_items_weighted(
                    liked_items=history_ids,
                    weights=weights,
                    item_emb=item_emb,
                    item_meta=item_meta,
                    topk=topk,
                )
                st.success("推荐结果：")
                for idx, r in enumerate(recs, 1):
                    st.markdown(
                        f"**[{idx}] {r['name']}**（相似度: {r['score']:.4f}）"
                    )
                    # if r["pictureURL"]:
                    #     st.image(r["pictureURL"], width=120)
                    if r["url"]:
                        st.markdown(f"- 链接：[LastFM]({r['url']})")
                    st.write("---")


if __name__ == "__main__":
    main()
