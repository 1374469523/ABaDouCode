# onehot 编码实现
import jieba
# 导入keras中的词汇映射器Tokenizer  分词器
from tensorflow.keras.preprocessing.text import Tokenizer
# 导入用于对象保存与加载的joblib
import joblib


def get_one_hot():
    # 准备语料库
    vocabs = {"周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗"}
    # 实例化Tokenizer
    my_tokenizer = Tokenizer()
    my_tokenizer.fit_on_texts(vocabs)
    print(my_tokenizer.index_word)
    print(my_tokenizer.word_index)
    for vocab in vocabs:
        zero_list = [0] * len(vocabs)
        idx = my_tokenizer.word_index[vocab] - 1
        zero_list[idx] = 1
        print(f'当前{vocab}的one-hot编码是{zero_list}')

    # 保存 tokenizer,方便下次使用
    mypath = "./myTokenizer"
    joblib.dump(my_tokenizer, mypath)
    print("模型保存成功")

def get_use_one_hot():
    load_my_tokenizer = joblib.load("./myTokenizer")
    print(load_my_tokenizer.word_index)
    print(load_my_tokenizer.index_word)

if __name__ == '__main__':
    # get_one_hot()
    get_use_one_hot()