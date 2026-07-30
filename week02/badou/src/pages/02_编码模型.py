"""
编码模型演示 — 语义编码 / 相似度计算 / 聚类可视化
===================================================
"""

import sys
from pathlib import Path

import streamlit as st
import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

st.set_page_config(page_title="编码模型", page_icon="🧬", layout="wide")

st.title("🧬 编码模型 (Embedding)")
st.markdown(
    "基于 **BAAI/bge-small-zh-v1.5** 将文本编码为 **512 维** 稠密向量，"
    "并演示向量在语义空间中的运算与可视化。"
)

# ── 模型信息 ─────────────────────────────────────────────────
st.caption("bge-small-zh-v1.5 ｜ 向量维度 512 ｜ 余弦相似度 ｜ 推理设备: CPU")

# ── 延迟加载 Embedding 模型 ────────────────────────────────
@st.cache_resource(show_spinner="⏳ 加载语义模型（首次较慢，约 10-20s）...")
def load_embedder():
    from src.embedding import EmbeddingService
    return EmbeddingService(device="cpu", show_progress=False)

emb = load_embedder()

tab1, tab2, tab3 = st.tabs([
    "📄 语义编码", "📐 相似度计算", "🎯 聚类可视化",
])


# ══════════════════════════════════════════════════════════════
# Tab 1：语义编码
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("将文本编码为稠密向量，观察语义向量空间的特性。")

    text_input = st.text_area(
        "输入文本（每行一条，分别编码）",
        value="人工智能正在改变世界\n"
              "今天天气真不错\n"
              "自然语言处理是 AI 的重要分支\n"
              "明天可能会下雨",
        height=120,
        key="encode_input",
    )

    if st.button("🚀 编码", key="run_encode", type="primary"):
        lines = [t.strip() for t in text_input.strip().split("\n") if t.strip()]

        if not lines:
            st.warning("请输入至少一行文本。")
        else:
            with st.spinner(f"编码 {len(lines)} 条文本..."):
                vecs = emb.encode(lines, normalize=True)

            st.success(f"编码完成！输出 shape: {vecs.shape}")

            st.markdown("##### 向量预览")
            for i, (text, vec) in enumerate(zip(lines, vecs)):
                with st.container(border=True):
                    st.markdown(f"**文本 {i + 1}:** {text}")
                    st.caption(f"向量（前 10 维 / 共 {len(vec)} 维）: "
                               f"`[{', '.join(f'{v:.4f}' for v in vec[:10])}, ...]`")

            # 向量统计
            st.markdown("##### 向量统计")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("维度", vecs.shape[1])
            with col2:
                st.metric("范数均值", f"{np.linalg.norm(vecs, axis=1).mean():.4f}")
            with col3:
                st.metric("标准差", f"{vecs.std():.4f}")
            with col4:
                st.metric("条数", vecs.shape[0])



# ══════════════════════════════════════════════════════════════
# Tab 2：相似度计算
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("计算文本间的语义相似度，支持 **一对多** 和 **矩阵** 两种模式。")

    sim_mode = st.radio("计算模式", ["一对多查询", "相似度矩阵"], horizontal=True)

    if sim_mode == "一对多查询":
        col_q, col_d = st.columns([1, 2])
        with col_q:
            query = st.text_input("查询文本", value="机器学习", key="sim_query")
        with col_d:
            docs = st.text_area(
                "候选文本（每行一条）",
                value="深度学习\n支持向量机\n今天的天气\n自然语言处理\n随机森林",
                height=130, key="sim_docs",
            )

        if st.button("🚀 计算相似度", key="run_sim_query", type="primary"):
            doc_list = [d.strip() for d in docs.strip().split("\n") if d.strip()]
            if query and doc_list:
                with st.spinner("计算中..."):
                    q_vec = emb.encode(query, normalize=True)
                    d_vecs = emb.encode(doc_list, normalize=True)
                    scores = (d_vecs @ q_vec).tolist()

                scored = sorted(zip(doc_list, scores), key=lambda x: -x[1])

                st.success(f"查询「{query}」与 {len(doc_list)} 条文本的相似度")

                for i, (doc, score) in enumerate(scored, 1):
                    max_score = max(s for _, s in scored)
                    pct = score / max_score if max_score > 0 else 0
                    st.markdown(
                        f"<div style='display:flex;align-items:center;"
                        f"padding:6px 0;border-bottom:1px solid #eee;'>"
                        f"<span style='min-width:28px;color:#888;'>{i}.</span>"
                        f"<span style='flex:1;'>{doc}</span>"
                        f"<span style='min-width:50px;text-align:right;"
                        f"font-weight:700;"
                        f"color:{'#ff4b4b' if score > 0.7 else '#ffa500' if score > 0.5 else '#888'};"
                        f"'>{score:.2%}</span>"
                        f"<progress value='{score}' max='1.0' "
                        f"style='width:80px;height:6px;border-radius:3px;"
                        f"margin-left:12px;'></progress>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    else:  # 相似度矩阵
        st.markdown("计算两组文本间的全部两两相似度。")

        col_a, col_b = st.columns(2)
        with col_a:
            group_a = st.text_area(
                "组 A（每行一条）",
                value="苹果\n香蕉\n电脑",
                height=120, key="mat_a",
            )
        with col_b:
            group_b = st.text_area(
                "组 B（每行一条）",
                value="水果\n电子产品\n天气\n编程",
                height=120, key="mat_b",
            )

        if st.button("🚀 计算相似度矩阵", key="run_sim_mat", type="primary"):
            list_a = [t.strip() for t in group_a.strip().split("\n") if t.strip()]
            list_b = [t.strip() for t in group_b.strip().split("\n") if t.strip()]

            if list_a and list_b:
                with st.spinner(f"计算 {len(list_a)}×{len(list_b)} 相似度矩阵..."):
                    sim_mat = emb.similarity(list_a, list_b)

                st.success(f"相似度矩阵 shape: {sim_mat.shape}")

                # 热力图
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(max(6, len(list_b) * 1.5),
                                                max(5, len(list_a) * 1.2)))
                im = ax.imshow(sim_mat, cmap="YlOrRd", vmin=0, vmax=1,
                               aspect="auto")

                ax.set_xticks(range(len(list_b)))
                ax.set_xticklabels(list_b, rotation=30, ha="right", fontsize=10)
                ax.set_yticks(range(len(list_a)))
                ax.set_yticklabels(list_a, fontsize=10)

                for i in range(len(list_a)):
                    for j in range(len(list_b)):
                        val = sim_mat[i, j]
                        ax.text(j, i, f"{val:.2f}",
                                ha="center", va="center",
                                fontsize=10,
                                color="white" if val > 0.5 else "black")

                ax.set_title("语义相似度热力图", fontsize=13)
                fig.colorbar(im, ax=ax, shrink=0.8)
                st.pyplot(fig)

                # 数据表格
                st.markdown("##### 详细数据")
                import pandas as pd
                df = pd.DataFrame(sim_mat, index=list_a, columns=list_b)
                st.dataframe(df.style.background_gradient(cmap="YlOrRd", vmin=0, vmax=1),
                             use_container_width=True)

    with st.expander("📝 相似度 API 说明"):
        st.code("""
# 余弦相似度（向量已 L2 归一化，点积 = 余弦相似度）
similarity = emb.similarity(texts_a, texts_b)
# → shape (len(texts_a), len(texts_b))

# 单条查询
q_vec = emb.encode("查询文本", normalize=True)
d_vecs = emb.encode(doc_list, normalize=True)
scores = d_vecs @ q_vec       # 点积
        """, language="python")


# ══════════════════════════════════════════════════════════════
# Tab 3：聚类可视化
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    将文本编码为向量后，使用 **KMeans 聚类** 并降维到 2D 平面进行可视化，
    直观观察语义空间中的类别分布。
    """)

    # 预设数据集
    cluster_preset = st.selectbox(
        "选择示例数据",
        ["📰 新闻标题混合", "🏷️ 电商商品分类", "✍️ 自定义输入"],
    )

    if cluster_preset == "📰 新闻标题混合":
        default_texts = (
            "苹果发布新一代iPhone手机\n"
            "华为推出新款折叠屏手机\n"
            "今日股市大涨突破3500点\n"
            "央行宣布降低存款准备金率\n"
            "中国女排横扫对手晋级决赛\n"
            "NBA总决赛即将开打\n"
            "ChatGPT推出全新语音功能\n"
            "百度发布文心一言大模型\n"
            "多地出台房地产调控新政\n"
            "北京房价环比小幅上涨"
        )

    elif cluster_preset == "🏷️ 电商商品分类":
        default_texts = (
            "Apple MacBook Pro笔记本电脑\n"
            "Dell 27寸4K显示器\n"
            "纯棉圆领T恤男装\n"
            "修身牛仔裤女款\n"
            "蓝牙无线降噪耳机\n"
            "智能手表运动版\n"
            "不粘锅三件套装\n"
            "真空保温杯500ml\n"
            "科幻小说三体全套\n"
            "Python编程从入门到实践"
        )

    else:
        default_texts = ""

    texts_input = st.text_area(
        "待聚类文本（每行一条）",
        value=default_texts,
        height=160, key="cluster_input",
    )

    col_k, col_seed, _ = st.columns([1, 1, 2])
    with col_k:
        n_clusters = st.number_input("聚类数 K", min_value=2, max_value=8, value=3, key="n_clusters")
    with col_seed:
        random_state = st.number_input("随机种子", min_value=0, max_value=9999, value=42, key="seed")

    run_cluster = st.button("🚀 聚类并可视化", key="run_cluster", type="primary")

    if run_cluster:
        texts = [t.strip() for t in texts_input.strip().split("\n") if t.strip()]

        if len(texts) < n_clusters:
            st.warning(f"文本条数 ({len(texts)}) 不能少于聚类数 ({n_clusters})。")
        else:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt

            with st.spinner("编码文本..."):
                vecs = emb.encode(texts, normalize=True)

            # 如果维度太高，先用 PCA 降维到 50 再给 t-SNE（避免过慢）
            with st.spinner("KMeans 聚类中..."):
                km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
                labels = km.fit_predict(vecs)

            with st.spinner("降维到 2D..."):
                if vecs.shape[1] > 50:
                    pca50 = PCA(n_components=min(50, vecs.shape[0], vecs.shape[1]),
                                random_state=random_state)
                    reduced = pca50.fit_transform(vecs)
                else:
                    reduced = vecs.copy()

                pca2 = PCA(n_components=2, random_state=random_state)
                coords = pca2.fit_transform(reduced)

            # 颜色映射
            colors = plt.colormaps.get_cmap("tab10")
            fig, ax = plt.subplots(figsize=(10, 7))

            for i in range(n_clusters):
                mask = labels == i
                pts = coords[mask]
                ax.scatter(pts[:, 0], pts[:, 1],
                           c=[colors(i)],
                           label=f"类别 {i + 1} ({mask.sum()} 条)",
                           s=120, alpha=0.8, edgecolors="white", linewidth=0.5)

                # 标注文本（为避免重叠，偏移一点）
                for j, idx in enumerate(np.where(mask)[0]):
                    ax.annotate(
                        texts[idx][:15] + ("…" if len(texts[idx]) > 15 else ""),
                        (pts[j, 0], pts[j, 1]),
                        fontsize=8, alpha=0.9,
                        xytext=(4, 4), textcoords="offset points",
                    )

            ax.set_title(f"文本聚类可视化（KMeans K={n_clusters} + PCA 降维）",
                         fontsize=14)
            ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.1%})")
            ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.1%})")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # 聚类结果表格
            st.markdown("##### 聚类结果")
            import pandas as pd
            result_df = pd.DataFrame({
                "文本": texts,
                "所属类别": [f"类别 {l + 1}" for l in labels],
            })

            # 每个类别展示样例
            for i in range(n_clusters):
                with st.expander(f"📁 类别 {i + 1}（{(labels == i).sum()} 条）"):
                    cluster_texts = [t for t, l in zip(texts, labels) if l == i]
                    for t in cluster_texts:
                        st.markdown(f"- {t}")

            # 聚类质量指标
            st.markdown("##### 聚类质量")
            from sklearn.metrics import silhouette_score
            if len(texts) > n_clusters and n_clusters > 1:
                try:
                    sil = silhouette_score(vecs, labels)
                    st.metric("轮廓系数 (Silhouette Score)", f"{sil:.4f}",
                              help="越接近 1 表示聚类效果越好；负数表示可能分错。")
                except Exception:
                    pass


