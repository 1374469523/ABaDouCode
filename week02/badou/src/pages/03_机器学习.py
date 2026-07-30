"""
机器学习演示 — sklearn 线性回归 / PyTorch 线性回归
====================================================
"""

import sys
from pathlib import Path

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 设置 matplotlib 中文字体
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="机器学习", page_icon="📈", layout="wide")

st.title("📈 机器学习演示")
st.markdown("scikit-learn 和 PyTorch 线性回归对比，观察梯度下降训练过程。")

tab1, tab2 = st.tabs(["🤖 sklearn 线性回归", "🔥 PyTorch 线性回归"])


# ══════════════════════════════════════════════════════════════
# Tab 1：sklearn 线性回归
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    ##### 使用 scikit-learn 的 `LinearRegression` 拟合线性数据

    数据生成公式：**y = 2x + 1 + noise**
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        n_samples = st.number_input("样本数", min_value=10, max_value=500, value=100, key="sk_n")
    with col2:
        noise = st.slider("噪声水平", 0.0, 3.0, 1.0, 0.1, key="sk_noise")
    with col3:
        run_sk = st.button("🚀 运行 sklearn", key="run_sk", type="primary")

    if run_sk:
        with st.spinner("训练中..."):
            # 生成数据
            np.random.seed(42)
            X = np.random.rand(n_samples, 1) * 10
            y_true = 2 * X + 1
            y = y_true + np.random.randn(n_samples, 1) * noise

            # 训练
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)

            a_learned = model.coef_[0, 0]
            b_learned = model.intercept_[0]
            y_pred = model.predict(X)

            # 评估
            from sklearn.metrics import r2_score, mean_squared_error
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)

        # 显示参数
        col_a, col_b, col_r2, col_mse = st.columns(4)
        with col_a:
            st.metric("斜率 a", f"{a_learned:.4f}", delta=f"{a_learned - 2:.4f}")
        with col_b:
            st.metric("截距 b", f"{b_learned:.4f}", delta=f"{b_learned - 1:.4f}")
        with col_r2:
            st.metric("R² 评分", f"{r2:.4f}")
        with col_mse:
            st.metric("均方误差 MSE", f"{mse:.4f}")

        # 绘图
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(X, y, alpha=0.6, label="数据点（含噪声）")
        ax.plot(np.sort(X, axis=0), y_true[np.argsort(X, axis=0).flatten()],
                "g--", label="真实: y=2x+1", linewidth=2)
        ax.plot(np.sort(X, axis=0), model.predict(np.sort(X, axis=0)),
                "r-", label=f"拟合: y={a_learned:.2f}x+{b_learned:.2f}", linewidth=2)
        ax.set_xlabel("X")
        ax.set_ylabel("y")
        ax.set_title("sklearn 线性回归")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)



# ══════════════════════════════════════════════════════════════
# Tab 2：PyTorch 线性回归
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    ##### 使用 PyTorch 的 `nn.Linear` + 梯度下降训练

    通过反向传播自动计算梯度，逐步优化参数。
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_samples_t = st.number_input("样本数", 10, 500, 100, key="torch_n")
    with col2:
        noise_t = st.slider("噪声水平", 0.0, 3.0, 1.0, 0.1, key="torch_noise")
    with col3:
        lr = st.select_slider("学习率 (lr)", options=[0.001, 0.005, 0.01, 0.05, 0.1], value=0.01, key="torch_lr")
    with col4:
        epochs = st.number_input("训练轮数", 50, 3000, 500, step=50, key="torch_epochs")

    run_torch = st.button("🚀 运行 PyTorch 训练", key="run_torch", type="primary")

    if run_torch:
        import torch
        import torch.nn as nn

        with st.spinner("训练中..."):
            # 生成数据
            np.random.seed(42)
            X_np = np.random.rand(n_samples_t, 1) * 10
            y_np = 2 * X_np + 1 + np.random.randn(n_samples_t, 1) * noise_t

            X_t = torch.from_numpy(X_np).float()
            y_t = torch.from_numpy(y_np).float()

            # 模型
            model = nn.Linear(1, 1)
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)

            # 训练记录
            loss_history = []
            param_history = []

            progress_bar = st.progress(0, text="训练中...")

            for epoch in range(epochs):
                y_pred = model(X_t)
                loss = loss_fn(y_pred, y_t)
                loss_history.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epoch % 50 == 0 or epoch == epochs - 1:
                    param_history.append({
                        "epoch": epoch + 1,
                        "weight": model.weight.item(),
                        "bias": model.bias.item(),
                        "loss": loss.item(),
                    })

                progress_bar.progress((epoch + 1) / epochs,
                                      text=f"Epoch {epoch + 1}/{epochs}  loss={loss.item():.4f}")

            progress_bar.empty()

        # 结果
        a_final = model.weight.item()
        b_final = model.bias.item()

        col_a, col_b, col_loss = st.columns(3)
        with col_a:
            st.metric("斜率 a", f"{a_final:.4f}", delta=f"{a_final - 2:.4f}")
        with col_b:
            st.metric("截距 b", f"{b_final:.4f}", delta=f"{b_final - 1:.4f}")
        with col_loss:
            st.metric("最终损失", f"{loss_history[-1]:.6f}")

        # 两张图
        fig_col1, fig_col2 = st.columns(2)

        with fig_col1:
            # 损失曲线
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.plot(loss_history, "b-", alpha=0.7, linewidth=1)
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss (MSE)")
            ax1.set_title("损失下降曲线")
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)

        with fig_col2:
            # 拟合结果
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            model.eval()
            with torch.no_grad():
                y_pred_np = model(X_t).numpy()

            ax2.scatter(X_np, y_np, alpha=0.6, label="数据点")
            sort_idx = np.argsort(X_np.flatten())
            ax2.plot(X_np[sort_idx], (2 * X_np + 1)[sort_idx],
                     "g--", label="真实: y=2x+1", linewidth=2)
            ax2.plot(X_np[sort_idx], y_pred_np[sort_idx],
                     "r-", label=f"拟合: y={a_final:.2f}x+{b_final:.2f}", linewidth=2)
            ax2.set_xlabel("X")
            ax2.set_ylabel("y")
            ax2.set_title("PyTorch 线性回归结果")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)

        # 参数变化表
        with st.expander("📊 训练过程参数变化"):
            import pandas as pd
            df = pd.DataFrame(param_history)
            df.columns = ["Epoch", "斜率 (Weight)", "截距 (Bias)", "Loss"]
            st.dataframe(df, use_container_width=True, hide_index=True)


