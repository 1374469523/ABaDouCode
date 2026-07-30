"""
sentence-transformers 加载 BGE 中文 embedding 模型并编码。
"""

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
# sentence_transformers 基于 torch 的库，加载编码模型；

# 本地模型路径（相对于本项目 asserts 目录）
_HERE = Path(__file__).resolve().parent
_DEFAULT_MODEL_PATH = str(_HERE.parent / "asserts" / "bge-small-zh-v1.5")

class EmbeddingService:
    """BGE embedding 编码服务。

    Args:
        model_path: 本地模型目录或 HuggingFace 模型名。
                    默认使用 ``asserts/bge-small-zh-v1.5/``。
        device: 推理设备 (``"cpu"`` / ``"cuda"`` / ``"mps"``)。
                为 ``None`` 时自动选择可用设备。
        show_progress: ``encode()`` 时是否显示进度条。
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        show_progress: bool = True,
    ):
        self.model_path = model_path or _DEFAULT_MODEL_PATH
        self.model = SentenceTransformer(
            self.model_path,
            device=device,
        )
        self._show_progress = show_progress

    def encode(
        self,
        sentences: str | list[str],
        batch_size: int = 64,
        normalize: bool = True,
        show_progress: bool | None = None,
    ) -> np.ndarray:
        """将文本编码为稠密向量。

        Args:
            sentences:  单条文本或文本列表。
            batch_size: 批量编码大小。
            normalize:  是否对输出向量做 L2 归一化。
                        开启后可直接用点积计算余弦相似度。
            show_progress: 是否显示进度条；默认使用构造时传入的值。

        Returns:
            shape 为 ``(n,)``（单条输入）或 ``(n, d)``（列表输入）的
            float32 向量，其中 ``d = 512``。
        """
        single = isinstance(sentences, str)
        inputs = [sentences] if single else sentences

        show = show_progress if show_progress is not None else self._show_progress

        embeddings = self.model.encode(
            inputs,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show,
        )

        return embeddings[0] if single else embeddings

    def similarity(
        self,
        a: str | list[str],
        b: str | list[str],
    ) -> np.ndarray:
        """计算两组文本间的余弦相似度矩阵。

        Args:
            a: 文本或文本列表。
            b: 文本或文本列表。

        Returns:
            shape ``(len(a), len(b))`` 的相似度矩阵。
        """
        vecs_a = self.encode(a, normalize=True)
        vecs_b = self.encode(b, normalize=True)

        if vecs_a.ndim == 1:
            vecs_a = vecs_a[np.newaxis, :]
        if vecs_b.ndim == 1:
            vecs_b = vecs_b[np.newaxis, :]

        return vecs_a @ vecs_b.T

if __name__ == "__main__":
    emb = EmbeddingService()

    # 编码
    docs = [
        "孟买上百万人挤贫民窟 首富1家5口住27层高楼豪",
        "一边倒！中国女排3-0横扫美国取七连胜 朱婷23",
        "横店演员住月租120块黑暗出租屋 到处都是蟑螂虫",
    ]
    vecs = emb.encode(docs, normalize=True)
    print(f"\n向量 shape: {vecs.shape}")       # (3, 512)
    print(f"向量[:5]:   {vecs[0, :5]}")        # 前 5 维

    # 相似度矩阵
    sim = emb.similarity("中国女排 朱婷", docs)
    print(f"\n相似度: {sim}")

    # 单条
    vec = emb.encode("台北101遇台风稳如泰山")
    print(f"\n单条向量 shape: {vec.shape}")     # (512,)
