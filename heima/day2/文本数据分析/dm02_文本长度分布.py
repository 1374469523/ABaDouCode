import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def dm_content_length():
    # 训练集
    train_tsv_data = pd.read_csv('./cn_data/train.tsv', sep='\t')
    train_tsv_data['sentence_length'] = train_tsv_data['sentence'].map(lambda x: len(x))
    sns.displot(x='sentence_length', data=train_tsv_data)

    # 读数据
    dev_csv_data = pd.read_csv('./cn_data/dev.tsv', sep='\t')
    # print(f'dev_csv_data.head()  --- > {dev_csv_data.head()}')
    dev_csv_data['sentence_length'] = dev_csv_data['sentence'].map(lambda x: len(x))
    print(f'dev_csv_data--》{dev_csv_data.head()}')
    # sns.countplot(x='sentence_length', data=dev_csv_data)
    # sns.displot(x='sentence_length', data=dev_csv_data, kind='kde')
    plt.xticks([])
    plt.title('train_data')
    plt.show()


def dm_sns_stripplot():
    train_tsv_data = pd.read_csv('./cn_data/train.tsv', sep='\t')

    train_tsv_data['sentence_length'] = train_tsv_data['sentence'].map(lambda x: len(x))
    sns.stripplot(y='sentence_length', x='label', data=train_tsv_data)
    plt.style.use('fivethirtyeight')
    plt.show()


if __name__ == '__main__':
    # dm_content_length()
    dm_sns_stripplot()
