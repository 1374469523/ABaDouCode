"""
FAQ 检索 — 全文检索 + 语义检索
================================
"""

import sys
from pathlib import Path

import streamlit as st

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import FaqDataset
from src.client import LLMClient
import jieba
import bm25s
from bm25s import BM25
import re

st.set_page_config(page_title="FAQ 检索", page_icon="🔍", layout="wide")

# ── 初始化 Session State ──────────────────────────────────
if "faq_ds" not in st.session_state:
    st.session_state.faq_ds = FaqDataset.from_json(
        _PROJECT_ROOT / "asserts" / "faq.json"
    )
    # 用 FAQ 问答文本构建 BM25 索引
    corpus = [
        f"{item.question} {item.answer}"
        for item in st.session_state.faq_ds.items
    ]
    corpus_tokens = [list(jieba.cut(t)) for t in corpus]
    bm25 = BM25()
    bm25.index(corpus_tokens, show_progress=False)
    st.session_state.bm25 = bm25

ds: FaqDataset = st.session_state.faq_ds

# ── 主标题 ─────────────────────────────────────────────────
st.title("🔍 FAQ 智能检索")
st.caption(f"共 {len(ds)} 条 FAQ")

# ── 两列布局 ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔎 全文检索（BM25）", "🧠 语义检索（向量相似度）", "🤖 LLM 分类",
])

# ══════════════════════════════════════════════════════════════
# Tab 1：全文检索（BM25）
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        "基于 **jieba 分词 + BM25 算法** 的相关性全文检索，"
        "根据词频和逆文档频率对结果排序，而非简单的布尔匹配。"
    )

    fulltext_query = st.text_input(
        "输入检索词",
        placeholder="例如: 退货退款",
        key="fulltext_query",
    )

    if fulltext_query:
        fulltext_query = fulltext_query.strip()
        with st.spinner("正在检索..."):
            try:
                query_tokens = [list(jieba.cut(fulltext_query))]
                results, scores = st.session_state.bm25.retrieve(
                    query_tokens, k=len(ds.items), show_progress=False
                )
            except Exception as e:
                st.error(f"检索出错：{e}")
                results, scores = [], []

        matched = []
        for i in range(results.shape[1]):
            doc_idx = results[0, i]
            score = float(scores[0, i])
            if score > 0:
                matched.append((doc_idx, score))

        if matched:
            st.success(f"✅ 找到 {len(matched)} 条结果")

            for rank, (doc_idx, score) in enumerate(matched, 1):
                item = ds.items[doc_idx]
                # 高亮查询词（简单实现）
                q = fulltext_query
                q_highlighted = item.question.replace(
                    q, f'<span style="color:red;font-weight:700">{q}</span>'
                )
                a_highlighted = item.answer.replace(
                    q, f'<span style="color:red;font-weight:700">{q}</span>'
                )
                with st.container(border=True):
                    cols = st.columns([4, 1])
                    with cols[0]:
                        st.markdown(
                            f"**Q{rank}:** {q_highlighted}",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"**A:** {a_highlighted}",
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"`{item.type}`")
                    with cols[1]:
                        st.markdown(
                            f"<div style='text-align:right'>"
                            f"<span style='font-size:1rem;font-weight:700;"
                            f"color:#ff4b4b'>{score:.3f}</span><br>"
                            f"<span style='font-size:0.8rem;color:#888'>BM25 分数</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
        else:
            st.info("未找到匹配结果，请尝试其他关键词。")
    else:
        # 无查询时展示当前筛选的 FAQ 列表
        st.markdown(f"##### 📋 当前 FAQ 列表（{len(ds)} 条）")
        for i, item in enumerate(ds.items, 1):
            with st.expander(f"**Q{i}:** {item.question}"):
                st.markdown(f"**类型：** `{item.type}`")
                st.markdown(f"**回答：** {item.answer}")


# ══════════════════════════════════════════════════════════════
# Tab 2：语义检索
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        "基于 **BGE-small-zh** 向量模型的语义检索，"
        "计算查询与 FAQ 的语义相似度。即使关键词不匹配也能找到相关内容。"
    )

    # 延迟加载 embedding 模型（首次使用时加载）
    if "emb_service" not in st.session_state:
        st.session_state.emb_service = None  # 占位
        st.session_state.faq_embeddings = None

    # 语义查询输入
    semantic_query = st.text_input(
        "输入查询内容",
        placeholder="例如: 如何退换货？",
        key="semantic_query",
    )

    top_k = st.slider("返回结果数量", min_value=3, max_value=20, value=5)

    if semantic_query and semantic_query.strip():
        query = semantic_query.strip()

        # 按当前筛选的 item 构建语义搜索
        search_items = ds.items

        # 首次使用需要加载 embedding 模型
        with st.spinner("加载语义模型（首次较慢）..."):
            if st.session_state.emb_service is None:
                from src.embedding import EmbeddingService
                st.session_state.emb_service = EmbeddingService(
                    device="cpu", show_progress=False
                )
            emb = st.session_state.emb_service

        with st.spinner("正在计算语义相似度..."):
            texts = [f"{item.question}" for item in search_items]
            try:
                query_vec = emb.encode(query, normalize=True)
                doc_vecs = emb.encode(texts, normalize=True)
                import numpy as np
                scores = (doc_vecs @ query_vec).tolist()
            except Exception as e:
                st.error(f"语义检索出错：{e}")
                scores = [0.0] * len(search_items)

        # 按分数排序
        scored = list(zip(search_items, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = [s for s in scored if s[1] > 0.3][:top_k]

        if scored:
            st.success(f"✅ 找到 {len(scored)} 条语义相关结果")

            # 分数条
            max_score = max(s[1] for s in scored) if scored else 1.0

            for i, (item, score) in enumerate(scored, 1):
                pct = score / max_score
                with st.container(border=True):
                    cols = st.columns([4, 1])
                    with cols[0]:
                        st.markdown(f"**Q{i}:** {item.question}")
                        st.markdown(f"**A:** {item.answer}")
                        st.markdown(f"`{item.type}`")
                    with cols[1]:
                        st.markdown(
                            f"<div style='text-align:right'>"
                            f"<span style='font-size:1.2rem;font-weight:700;"
                            f"color:{'#ff4b4b' if pct > 0.8 else '#ffa500' if pct > 0.6 else '#888'}'>"
                            f"{score:.2%}</span><br>"
                            f"<progress value='{score}' max='1.0' "
                            f"style='width:100%;height:6px;border-radius:3px;'></progress>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
        else:
            st.info("未找到语义相似度 > 0.3 的结果，请尝试其他查询。")
    else:
        # 无查询时提示
        st.info("👆 输入查询内容后点击回车进行语义检索。")

    # 模型信息
    with st.expander("ℹ️ 关于语义检索模型"):
        st.markdown("""
        - **模型**: BAAI/bge-small-zh-v1.5
        - **向量维度**: 512
        - **距离度量**: 余弦相似度（L2 归一化后点积）
        - **特点**: 轻量级中文 embedding 模型，适合本地部署
        """)


# ══════════════════════════════════════════════════════════════
# Tab 3：LLM 分类
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    使用 LLM 对用户提问进行分类，识别属于哪个 FAQ 类型，
    并可基于分类结果快速查看对应类型的 FAQ。
    """)

    if "llm" not in st.session_state:
        st.session_state.llm = LLMClient()
    llm = st.session_state.llm

    classify_input = st.text_area(
        "输入用户提问",
        value="我买的手机屏幕碎了，能免费换吗？",
        height=80, key="classify_input",
    )

    classify_btn = st.button("🤖 LLM 分类", key="classify_btn", type="primary")

    if classify_btn and classify_input.strip():
        text = classify_input.strip()
        types_str = "、".join(sorted(ds.types))

        with st.spinner("LLM 分类中..."):
            try:
                resp = llm.chat([
                    {
                        "role": "system",
                        "content": (
                            f"你是电商客服分类助手。用户的提问属于以下类型之一：{types_str}。"
                            f"请判断类型并输出 JSON：{{\"type\": \"类型名\", \"reason\": \"判断理由\"}}"
                        ),
                    },
                    {"role": "user", "content": text},
                ])

                import json as _json
                parsed = None
                try:
                    parsed = _json.loads(resp)
                except Exception:
                    import re as _re
                    m = _re.search(r'"type"\s*:\s*"([^"]+)"', resp)
                    if m:
                        found = m.group(1)
                        if found in ds.types:
                            parsed = {"type": found, "reason": "正则提取"}
                        else:
                            for t in ds.types:
                                if t in resp:
                                    parsed = {"type": t, "reason": "关键词匹配"}
                                    break

                if parsed and parsed.get("type") in ds.types:
                    cls_type = parsed["type"]
                    reason = parsed.get("reason", "")
                    st.success(f"**分类结果：`{cls_type}`**")
                    if reason:
                        st.caption(f"理由：{reason}")

                    # 展示该类型下的 FAQ
                    related = ds.filter_by_type(cls_type)
                    st.markdown(f"**相关 FAQ（{len(related)} 条）：**")
                    for j, item in enumerate(related, 1):
                        with st.expander(f"**Q{j}:** {item.question}"):
                            st.markdown(f"**回答：** {item.answer}")

                    # 一键跳转 BM25 搜索（自动填充）
                    st.markdown("---")
                    st.markdown("直接在 FAQ 中搜索：")
                    bm25_q = classify_input.strip()
                    # 用 session_state 跨 tab 传值不可行，用 st.link 或提示
                    st.info(f"💡 切到「全文检索」Tab 输入「{bm25_q}」查看 BM25 排序结果。")
                else:
                    st.warning(f"LLM 返回内容无法解析：{resp[:200]}")

            except Exception as e:
                st.error(f"LLM 调用失败：{e}")

    elif not classify_input.strip():
        st.info("👆 输入提问内容后点击「LLM 分类」。")
