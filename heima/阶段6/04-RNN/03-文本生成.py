import jieba
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn


# 构建词表
def build_vocab():
    all_words = []
    unique_words = []
    # 1.读取数据
    for line in open('/Users/mac/Desktop/AI20深度学习/02-code/04-RNN/data/jaychou_lyrics.txt', 'r'):
        # 2.分词
        words = jieba.lcut(line)
        all_words.append(words)
        # print(all_words)
        # 3.去重
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
        # print(unique_words)
        # break
    # 4.构建字典
    word2index = {word: i for i, word in enumerate(unique_words)}
    print(word2index)
    # 5.文本转id
    print(all_words)
    corpus_id = []
    for words in all_words:
        temp = []
        for word in words:
            temp.append(word2index[word])
        temp.append(word2index[' '])
        corpus_id.extend(temp)
    return unique_words, word2index, len(unique_words), corpus_id


# 构建数据集
class LyricsDataset(Dataset):
    def __init__(self, corpus_id, num_char):
        self.corpus_id = corpus_id
        self.num_char = num_char
        self.word_count = len(self.corpus_id)
        self.num = len(self.corpus_id) // self.num_char

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        start = min(max(idx, 0), self.word_count - self.num_char - 2)
        x = self.corpus_id[start:start + self.num_char]
        y = self.corpus_id[start + 1:start + 1 + self.num_char]
        return torch.tensor(x), torch.tensor(y)


# 模型构建
class TextGenerator(nn.Module):
    def __init__(self, word_count):
        super(TextGenerator, self).__init__()
        self.embed = nn.Embedding(num_embeddings=word_count, embedding_dim=128)
        self.rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=1)
        self.out = nn.Linear(in_features=256, out_features=word_count)

    def forward(self, inputs, hidden):
        embeds = self.embed(inputs)
        out, hid = self.rnn(embeds.transpose(0, 1), hidden)
        out = self.out(out.reshape(-1, 256))
        return out

    def init_hidden(self, bs):
        return torch.zeros(1, bs, 256)


# 模型训练
def train(dataset, model):
    # 损失
    cri = nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=[0.9, 0.99])
    # 遍历
    epoches = 10
    for eopch in range(epoches):
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        loss_sum = 0
        sample=0.001
        for x, y in dataloader:
            h0 = model.init_hidden(bs=2)
            out = model(x, h0)
            y =torch.transpose(y,0,1).contiguous().view(-1)
            loss =cri(out,y)
            loss_sum+=loss.item()
            sample+=1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            break
        print(loss_sum/sample)
    torch.save(model.state_dict(),'model.pth')
# 模型预测
def predict(model,start_word,len,unique_words,word2index):
    model.load_state_dict(torch.load('model.pth'))
    wor_index = word2index[start_word]
    h0=model.init_hidden(bs=1)
    words_list = []
    for _ in range(len):
        out =model(torch.tensor([[wor_index]]),h0)
        wor_index = torch.argmax(out)
        words_list.append(unique_words[wor_index])
    for word in words_list:
        print(word,end='')



if __name__ == '__main__':
    unique_words, word2index, word_count, corpus_id = build_vocab()
    dataset = LyricsDataset(corpus_id, 10)
    print(dataset[0])
    model = TextGenerator(word_count)
    # print(model.parameters())
    # train(dataset, model)
    predict(model,'青春',50,unique_words,word2index)
