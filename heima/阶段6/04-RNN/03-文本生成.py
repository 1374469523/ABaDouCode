# ============================================================
# 演示主题：用 RNN 做文本生成（"预测下一个字"）
#   核心思想：把"预测下一个字"当成一个多分类问题——
#   词表里的每个字都是一个"类别"，模型对每个位置输出所有字的得分（logits），
#   取得分最高（概率最大）的字作为"下一个字"，反复生成、拼成一句话。
#
# 整体流程（可对照下面的函数阅读）：
#   1. 数据准备：读取周杰伦歌词 -> jieba 分词 -> 去重得到"词表"
#   2. 映射：word2index 把"词"转成"数字索引"，再把整篇文本转成数字序列 corpus_id
#   3. 数据集：用滑动窗口切样本——用前 num_char 个字预测后 num_char 个字
#   4. 模型：Embedding(词->稠密向量) -> RNN(按时间步处理) -> Linear(映射回词表得分)
#   5. 训练：CrossEntropyLoss 多分类损失，让模型学会"看到前面的字，猜下一个字"
#   6. 预测：给定起始词，循环采样"下一个字"，拼出新的歌词
# ============================================================
import jieba
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn


# 构建词表
def build_vocab():
    all_words = []
    unique_words = []
    # 1.读取数据
    # 注意：这里写死的是 Mac 上的绝对路径，运行时需改成自己机器上
    #      "data/jaychou_lyrics.txt" 的实际绝对路径
    for line in open('/Users/mac/Desktop/AI20深度学习/02-code/04-RNN/data/jaychou_lyrics.txt', 'r'):
        # 2.分词
        # 把这一行歌词按词切开，返回词的列表
        words = jieba.lcut(line)
        # all_words 收集每一行分词后的结果（外层按"行"，内层是该行的词列表）
        all_words.append(words)
        # print(all_words)
        # 3.去重
        # 用"列表 + not in"去重：保证每个词在 unique_words 里只出现一次，
        # 且保持"首次出现"的先后顺序（这对后面词表索引的稳定性很重要）
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
        # print(unique_words)
        # break
    # 4.构建字典
    # word2index：词 -> 索引 的映射，即给每个词一个唯一的数字编号。
    #   有了它，就能把"中文词"变成"数字"，因为神经网络只能处理数字
    word2index = {word: i for i, word in enumerate(unique_words)}
    print(word2index)
    # 5.文本转id
    # 把整篇歌词（所有行）展平成一长串数字序列 corpus_id
    print(all_words)
    corpus_id = []
    for words in all_words:
        temp = []
        for word in words:
            temp.append(word2index[word])   # 每个词 -> 它的数字索引
        temp.append(word2index[' '])        # 每行歌词结尾补一个"空格"索引，作为行与行之间的分隔符
        corpus_id.extend(temp)              # 把当前行的数字追加到整篇序列后面
    # 返回：词表(去重后的词列表)、词->索引字典、词表大小、整篇歌词的数字序列
    return unique_words, word2index, len(unique_words), corpus_id


# 构建数据集
# 把一长串数字序列切成很多个"(输入, 标签)"训练样本：
#   输入 = 连续的 num_char 个字；标签 = 往后错一位的同样长度个字（即"下一个字"）
class LyricsDataset(Dataset):
    def __init__(self, corpus_id, num_char):
        self.corpus_id = corpus_id        # 整篇歌词的数字序列
        self.num_char = num_char          # 窗口大小：用几个字预测几个字
        self.word_count = len(self.corpus_id)        # 整篇文本总共多少个字
        self.num = len(self.corpus_id) // self.num_char   # 能切出多少个样本（向下取整）

    def __len__(self):
        # DataLoader 会调用它来知道"一共多少条数据"
        return self.num

    def __getitem__(self, idx):
        # 找到本样本的起始位置；min/max 保证不越界（索引不能小于 0，也不能太靠后）
        start = min(max(idx, 0), self.word_count - self.num_char - 2)
        x = self.corpus_id[start:start + self.num_char]          # 输入：连续 num_char 个字
        y = self.corpus_id[start + 1:start + 1 + self.num_char]  # 标签：往后错一位的 num_char 个字
        return torch.tensor(x), torch.tensor(y)                  # 转成 tensor，返回 (输入, 标签)


# 模型构建
class TextGenerator(nn.Module):
    def __init__(self, word_count):
        super(TextGenerator, self).__init__()
        # 词嵌入层：词表大小 word_count 个词，每个词映射成 128 维的稠密向量
        #   输入形状 (batch, seq_len) 的词索引 -> 输出形状 (batch, seq_len, 128)
        self.embed = nn.Embedding(num_embeddings=word_count, embedding_dim=128)
        # RNN 层：
        #   input_size=128   -> 每步输入一个 128 维词向量
        #   hidden_size=256  -> 隐藏状态（记忆）的维度
        #   num_layers=1     -> 单层
        #   注意：未设置 batch_first=True，所以输入要按 (seq_len, batch, 128) 来给
        self.rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=1)
        # 全连接层：把 RNN 输出的 256 维隐藏态，映射回"词表大小"的维度。
        #   输出就是每个位置对所有候选词的得分（logits），得分最高的词就是"预测的下一个字"
        self.out = nn.Linear(in_features=256, out_features=word_count)

    def forward(self, inputs, hidden):
        # inputs 形状 (batch, seq_len)：一批样本里每个位置的词索引
        embeds = self.embed(inputs)                 # -> (batch, seq_len, 128) 词向量
        # RNN 默认 batch_first=False，需要把维度调成 (seq_len, batch, 128)，
        # 所以 transpose(0, 1) 交换前两维；hidden 是初始隐藏状态 h0
        out, hid = self.rnn(embeds.transpose(0, 1), hidden)
        # out 形状 (seq_len, batch, 256)；reshape(-1, 256) 把"每个位置"拍平成一行，
        # 经过 Linear 后每行 = 该位置对所有候选词的得分
        out = self.out(out.reshape(-1, 256))
        return out

    def init_hidden(self, bs):
        # 生成初始隐藏状态 h0：形状 (num_layers, batch, hidden_size) = (1, bs, 256)
        #   全 0 表示从"空白记忆"开始处理序列
        return torch.zeros(1, bs, 256)


# 模型训练
def train(dataset, model):
    # 损失
    # 多分类交叉熵损失。input 期望形状 (N, C)（N 个样本，C 个类别，C=词表大小），
    #   即"每个位置对所有词的得分"；label 是每个样本正确类别（正确词）的索引
    cri = nn.CrossEntropyLoss()
    # 优化器
    # Adam 优化器：lr=0.0001 学习率；betas 是 Adam 的两个动量系数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=[0.9, 0.99])
    # 遍历
    epoches = 10   # 整个数据集训练 10 轮
    for eopch in range(epoches):
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)   # 每次取 2 条样本，并打乱顺序
        loss_sum = 0       # 累计本轮损失，用于打印平均损失
        sample=0.001       # 计数（初始给个小值避免除 0，也让平均损失数字不至于太大）
        for x, y in dataloader:
            h0 = model.init_hidden(bs=2)      # 每个 batch 开始时重新初始化隐藏状态
            out = model(x, h0)                # 前向传播，out 形状 (batch*seq_len, word_count)
            # 标签 y 原本形状 (batch, seq_len)；转置 + view(-1) 展平成一维，
            # 让 y 的每个位置与 out 的每一行一一对应
            y =torch.transpose(y,0,1).contiguous().view(-1)
            loss =cri(out,y)                  # 计算交叉熵损失
            loss_sum+=loss.item()             # 累计损失（item() 取出数值）
            sample+=1                         # 样本计数 +1
            optimizer.zero_grad()             # 清空上一步计算的梯度
            loss.backward()                   # 反向传播，计算各参数的梯度
            optimizer.step()                  # 用梯度更新参数
            break                             # 只跑一个 batch 就跳出（演示代码，为了快速看到效果）
        print(loss_sum/sample)                # 打印本轮平均损失
    torch.save(model.state_dict(),'model.pth')   # 保存训练好的模型参数到 model.pth
# 模型预测
def predict(model,start_word,len,unique_words,word2index):
    model.load_state_dict(torch.load('model.pth'))   # 加载训练好的模型参数
    wor_index = word2index[start_word]   # 把起始词转成索引，作为第一个字
    h0=model.init_hidden(bs=1)           # 初始隐藏状态，bs=1（只生成一句）
    words_list = []
    for _ in range(len):                 # 循环生成 len 个字
        # 输入形状 (1, 1) = (batch=1, seq_len=1)：每次只喂"当前最后一个字"的索引
        out =model(torch.tensor([[wor_index]]),h0)
        # torch.argmax 取"得分最高"的类别索引，即概率最高的"下一个字"
        wor_index = torch.argmax(out)
        words_list.append(unique_words[wor_index])   # 把索引翻译回词，收集起来
    for word in words_list:
        print(word,end='')   # 不换行地逐个打印，拼成一句歌词



if __name__ == '__main__':
    unique_words, word2index, word_count, corpus_id = build_vocab()   # 第一步：构建词表
    dataset = LyricsDataset(corpus_id, 10)   # 第二步：构建数据集，窗口大小 10（用 10 个字预测后 10 个字）
    print(dataset[0])                        # 打印第一个训练样本 (输入, 标签)
    model = TextGenerator(word_count)        # 第三步：创建模型，word_count = 词表大小
    # print(model.parameters())
    # train(dataset, model)                    # 第四步：训练（默认注释掉，演示时可取消注释执行）
    predict(model,'青春',50,unique_words,word2index)   # 第五步：以"青春"为开头，自动生成 50 个字
