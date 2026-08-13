import jieba
# 导入chain方法用于扁平化列表
from itertools import chain
import pandas as pd


def get_vocabs():
    csv_data = pd.read_csv("cn_data/train.tsv", sep='\t')
    sentences = csv_data['sentence']
    csv_data_vocabs = set(chain(*map(lambda x: jieba.lcut(x), sentences)))

# 进行训练集的句子进行分词, 并统计出不同词汇的总数
