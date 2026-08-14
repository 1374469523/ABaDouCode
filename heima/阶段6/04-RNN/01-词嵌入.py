# ============================================================
# 演示主题：词嵌入（Word Embedding）
#   用 nn.Embedding 把"词"映射成稠密向量，体会它与 one-hot 编码的区别。
#
# 核心结论：
#   1. one-hot 编码的缺点：向量维度 = 词表大小（几千上万维）、绝大部分是 0、
#      任意两个词的向量互相垂直、距离都一样，表达不了词与词之间的语义关系。
#   2. 词嵌入（Embedding）的改进：把每个词映射成一个小维度（如 128 维）的
#      "稠密向量"，向量里每个位置都有数值；意思相近的词，它们的向量也更接近。
#   3. nn.Embedding 本质是一张"可学习的查表"：输入词索引 -> 输出对应的向量，
#      训练过程中这张表会不断被优化（即词向量本身是学出来的）。
# ============================================================
import jieba
import torch
import torch.nn as nn

# 分词：jieba.lcut 把一句话按词切开，返回词的列表
text = '北京冬奥的进度条已经过半，不少外国运动员在完成自己的比赛后踏上归途。'
words =jieba.lcut(text)
# print(words)

# 去重：set 去掉重复的词，再转回 list（顺序会变化，这里不关心顺序）
un_words=list(set(words))
# print(un_words)
# 词表大小 = 去重后一共有多少个不同的词
num = len(un_words)
# print(num)
# 调用embeding
# 创建 Embedding 层（词嵌入层）：
#   num_embeddings=num  -> 词表大小，即一共有多少种不同的词（决定"查表"有多少行）
#   embedding_dim=3     -> 每个词映射成几维向量（这里用 3 维只是为了演示，实际常用 128/256）
#   底层原理：相当于一张形状为 (num, 3) 的矩阵，用"词的索引"去取对应的那一行，作为该词的向量
embeds =nn.Embedding(num_embeddings=num,embedding_dim=3)
# print(embeds(torch.tensor(5)))

for i,word in enumerate(un_words):
    print(word)
    # embeds(torch.tensor(i))：把"词的索引 i"转成 tensor 后传入 Embedding，
    # 返回该词对应的 3 维向量（一个形状为 [3] 的 tensor）
    print(embeds(torch.tensor(i)))
