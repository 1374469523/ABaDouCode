"""
Streamlit 多页应用 — 主入口
============================
运行: streamlit run src/app.py
"""

import streamlit as st
from pathlib import Path
import sys

# 确保项目根目录在 sys.path 中
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import load_default_faq

st.set_page_config(
    page_title="AI 学习工具箱",
    page_icon="🧰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 加载 FAQ 统计 ──────────────────────────────────────────
faq_ds = load_default_faq()
total_faq = len(faq_ds)
faq_types = sorted(faq_ds.types)


# ── 样式 ───────────────────────────────────────────────────
st.markdown("""
<style>
    .card {
        background: #f0f2f6;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
        transition: box-shadow 0.2s;
    }
    .card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #ff4b4b;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

# ── 标题 ───────────────────────────────────────────────────
st.title("🧰 AI 学习工具箱")
st.markdown("""
本应用汇集了本项目的多个模块演示，涵盖 **倒排索引**、**编码模型**、
**FAQ 检索**和 **机器学习** 四大主题。
""")

# ── 统计卡片行 ────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f'<div class="card"><div class="stat-number">{total_faq}</div><div class="stat-label">FAQ 条目</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="card"><div class="stat-number">{len(faq_types)}</div><div class="stat-label">FAQ 类型</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="card"><div class="stat-number">4</div><div class="stat-label">页面总数</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="card"><div class="stat-number">512</div><div class="stat-label">向量维度</div></div>', unsafe_allow_html=True)
with col5:
    st.markdown(f'<div class="card"><div class="stat-number">3</div><div class="stat-label">ML 模型</div></div>', unsafe_allow_html=True)

# ── 导航卡片 ──────────────────────────────────────────────
st.subheader("📂 功能导航")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    with st.container(border=True):
        st.markdown("### 🔎 倒排索引")
        st.markdown("中文分词 + 布尔检索（AND / OR / NOT），"
                    "在新闻标题数据集上体验全文搜索引擎的核心原理。")

    with st.container(border=True):
        st.markdown("### 🧬 编码模型")
        st.markdown("BGE 中文语义编码、**相似度计算**、KMeans **聚类可视化**，"
                    "直观感受向量空间的语义分布。")

with nav_col2:
    with st.container(border=True):
        st.markdown("### 🔍 FAQ 检索")
        st.markdown(f"**{total_faq}** 条电商 FAQ，支持**全文检索** + **语义检索**，"
                    f"涵盖 {', '.join(f'`{t}`' for t in faq_types)} 等类型。")

with nav_col3:
    with st.container(border=True):
        st.markdown("### 📈 机器学习")
        st.markdown("scikit-learn 线性回归、PyTorch 线性回归和全连接网络的"
                    "训练过程可视化。")

# ── 底部提示 ──────────────────────────────────────────────
st.divider()
st.info("👈 使用左侧侧边栏导航切换不同功能页面。")
