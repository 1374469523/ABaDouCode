import fasttext


# 训练词向量模型
def dm_fasttext_train_save_load():
    # 1、使用train_unsupervised(无监督训练方法) 训练词向量
    fasttext_train_unsupervised = fasttext.train_unsupervised('./data/fil9')
    print('训练词向量 ok')
    # 2、保存模型
    fasttext_train_unsupervised.save_model('./data/fil9_unsupervised')


# 获取某个词的词向量和检验模型效果
def dm_fasttext_train_save_load_model():
    fasttext_load_model_unsupervised = fasttext.load_model("data/fil9_unsupervised")
    # 直接获取某个词的向量
    results = fasttext_load_model_unsupervised.get_word_vector("the")
    print(type(results))
    print(results.shape)
    print(results)

    get_nearest_neighbors = fasttext_load_model_unsupervised.get_nearest_neighbors("dog")
    print(f'dog的临近词-->{get_nearest_neighbors}')

# 训练词向量模型:修改参数
def dm_fasttext_03():
    # 直接开始训练：以非监督的方式进行
    model = fasttext.train_unsupervised('./data/ai20aa',"cbow", dim=100, lr=0.1, epoch=1)
    # 保存模型
    model.save_model('./data/fil9_unsupervised_new.bin')

if __name__ == '__main__':
    # dm_fasttext_train_save_load()
    dm_fasttext_train_save_load_model()
