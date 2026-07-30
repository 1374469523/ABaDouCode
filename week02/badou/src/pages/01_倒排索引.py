
"""
倒排索引演示 — 中文分词 + 布尔检索
=====================================
在新闻标题数据集上体验全文搜索引擎的核心原理。
"""

import sys
from pathlib import Path

import streamlit as st

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.invert_index import InvertedIndex

st.set_page_config(page_title="倒排索引", page_icon="🔎", layout="wide")

# ── 初始化 ──────────────────────────────────────────────────
if "news_idx" not in st.session_state:
    news_file = _PROJECT_ROOT / "asserts" / "爬虫-新闻标题.txt"
    with st.spinner("正在构建倒排索引（分词中）..."):
        idx = InvertedIndex()
        idx.load_docs(str(news_file))
        st.session_state.news_idx = idx
        st.session_state.news_count = len(idx.doc_list)

idx: InvertedIndex = st.session_state.news_idx
st.title("🔎 倒排索引搜索引擎")
st.caption(
    f"数据集：{st.session_state.news_count} 条新闻标题  |  "
    f"词项数：{len(idx.index)} 个"
)

# ── 索引统计 ────────────────────────────────────────────────
with st.expander("📊 索引统计", expanded=False):
    # 按文档频率排序的词项
    term_doc_counts = [(term, len(postings)) for term, postings in idx.index.items()]
    term_doc_counts.sort(key=lambda x: -x[1])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**出现最多的词项（Top 20）：**")
        top_df_data = {
            "词项": [t for t, _ in term_doc_counts[:20]],
            "包含文档数": [c for _, c in term_doc_counts[:20]],
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(top_df_data), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**总览：**")
        st.markdown(f"- 文档总数：**{st.session_state.news_count}**")
        st.markdown(f"- 不同词项数：**{len(idx.index)}**")
        st.markdown(f"- 唯一词项占比：**{len(idx.index) / st.session_state.news_count:.1%}**")
        st.markdown(f"- 平均每条标题词数：**{sum(len(t.split()) for t in idx.doc_list) / len(idx.doc_list):.1f}**")

        # 文档长度分布
        import jieba
        doc_lengths = [len(list(jieba.cut(d))) for d in idx.doc_list]
        st.markdown(f"- 最短标题：**{min(doc_lengths)}** 词")
        st.markdown(f"- 最长标题：**{max(doc_lengths)}** 词")

    # Top terms bar chart
    import matplotlib.pyplot as plt
    import numpy as np

    top_terms = term_doc_counts[:15]
    fig, ax = plt.subplots(figsize=(8, 4))
    terms = [t for t, _ in top_terms]
    counts = [c for _, c in top_terms]
    ax.barh(range(len(terms)), counts, color="#4CAF50")
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms)
    ax.set_xlabel("包含文档数")
    ax.set_title("高频词项 Top 15")
    ax.invert_yaxis()
    st.pyplot(fig)


# ── 布尔查询 ────────────────────────────────────────────────
st.markdown("##### 🔍 布尔查询")

query_col, help_col = st.columns([3, 1])
with query_col:
    query = st.text_input(
        "输入布尔查询（支持 AND / OR / NOT）",
        placeholder="例如: 苹果 AND (芯片 OR 高通)",
        key="bool_query",
    )
with help_col:
    st.markdown("""
    | 操作符 | 示例 |
    |--------|------|
    | **AND** | `A AND B` |
    | **OR**  | `A OR B` |
    | **NOT** | `A NOT B` |
    | **()**  | `(A OR B) AND C` |
    """)

if query:
    query = query.strip()
    with st.spinner("检索中..."):
        try:
            results = idx.search(query, highlight=True)
        except Exception as e:
            st.error(f"查询语法错误：{e}")
            results = []

    if results:
        st.success(f"✅ 找到 {len(results)} 条结果")

        # 显示结果
        for i, doc_html in enumerate(results, 1):
            st.markdown(
                f"<div style='padding:6px 0;border-bottom:1px solid #eee;'>"
                f"<span style='color:#888;'>{i}.</span> {doc_html}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("未找到匹配结果。")

    # 显示转化后的表达式
    with st.expander("📝 查看转化后的集合运算表达式"):
        expr = idx.conv_query(query)
        st.code(expr, language="python")
        st.markdown("（内部将布尔查询转化为 Python 集合运算后求值）")

else:
    # 无查询时展示样例数据
    st.markdown("##### 📰 新闻标题样例")
    st.markdown("输入查询开始检索，以下是数据集中部分新闻标题：")

    sample_docs = idx.doc_list[:10]
    for i, doc in enumerate(sample_docs, 1):
        st.markdown(f"{i}. {doc}")

    st.markdown(f"... 共 {st.session_state.news_count} 条")


# ── 词项查询 ────────────────────────────────────────────────
st.divider()
st.markdown("##### 🔬 词项查询")

term = st.text_input("查看某个词的倒排列表", placeholder="例如: 苹果", key="term_query")

if term:
    term = term.strip()
    postings = idx.index.get(term, set())
    if postings:
        doc_count = len(postings)
        st.markdown(
            f"词项 **`{term}`** 出现在 **{doc_count}** 篇文档中 "
            f"(占比 {doc_count/st.session_state.news_count:.1%})"
        )

        with st.expander(f"查看所有包含「{term}」的文档"):
            for doc_id in sorted(postings):
                highlighted = idx.highlighter(idx.doc_list[doc_id], term)
                st.markdown(f"- ID {doc_id}: {highlighted}", unsafe_allow_html=True)
    else:
        st.info(f"词项「{term}」未出现在索引中。")


# ── 代码参考 ────────────────────────────────────────────────

