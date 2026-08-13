import jieba
import torch
import torch.nn as nn

# 分词
text = '北京冬奥的进度条已经过半，不少外国运动员在完成自己的比赛后踏上归途。'
words =jieba.lcut(text)
# print(words)

# 去重
un_words=list(set(words))
# print(un_words)
num = len(un_words)
# print(num)
# 调用embeding
embeds =nn.Embedding(num_embeddings=num,embedding_dim=3)
# print(embeds(torch.tensor(5)))

for i,word in enumerate(un_words):
    print(word)
    print(embeds(torch.tensor(i)))
