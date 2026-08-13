import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# 思路分析 : 获取标签数量分布
# 0 什么标签数量分布：求标签0有多少个 标签1有多少个 标签2有多少个
# 1 设置显示风格plt.style.use('fivethirtyeight')
# 2 pd.read_csv(path, sep='\t') 读训练集 验证集数据
# 3 sns.countplot() 统计label标签的0、1分组数量
# 4 画图展示 plt.title() plt.show()
# 注意1：sns.countplot()相当于select * from tab1 group by
def dm_label_sns_countplot():
    # plt.style.use('fivethirtyeight')
    # # 直接获取数据
    # train_data = pd.read_csv('cn_data/train.tsv', sep='\t')
    # print(f'train_data-->{train_data.head()}')
    # # 统计标签数量分布
    # sns.countplot(x='label', data=train_data)
    # plt.title("train_data")
    # plt.show()
    train_data_dev = pd.read_csv('cn_data/dev.tsv', sep='\t')
    print(f'train_data_dev--=->{train_data_dev.head()}')
    print(f'dev_data--》{train_data_dev.head()}')
    # 统计标签数量分布
    # sns.countplot(x='label', data=dev_data)
    # x和y只能写一个，要不是x轴显示要不是y轴显示，hue按照哪类分组

    sns.countplot(x='label', data=train_data_dev,hue='label')
    plt.show()

if __name__ == '__main__':
    dm_label_sns_countplot()
