import jieba
import jieba.posseg as pesg
from jieba.posseg import pair

content = "汤谷智能是一家伟大的EDA公司，有很多人在这个地方工作，比如李某杰"


def jieba1():
    jieba_cut = jieba.cut(content, cut_all=False, HMM=True)
    print(jieba_cut)
    # 返回一个生成器
    # <generator object Tokenizer.cut at 0x7f8d9053e650>
    # 从 generator object 里面取元素

    # print(next(jieba_cut))
    # print(next(jieba_cut))
    # print(next(jieba_cut))
    # print(next(jieba_cut))
    # print(next(jieba_cut))
    # print(next(jieba_cut))
    # print(next(jieba_cut))
    # print(next(jieba_cut))

    # for cut in jieba_cut:
    #     print(cut)
    # 强制把对象变为列表
    # print(list(jieba_cut))
    lcut = jieba.lcut(content, cut_all=False)  # 返回的直接是一个list "l"代表的就是list
    print(lcut)


# 全模式分词
def jieba2():
    jieba_cut = jieba.cut(content, cut_all=True, HMM=True)
    print(list(jieba_cut))


# 搜索引擎分词
def jieba3():
    jieba_cut = jieba.lcut_for_search(content)
    print(jieba_cut)


# 繁体字和用户自定义词典
def jieba4():
    content = "李某杰烦恼即是菩提,我暫且不提"
    jieba.load_userdict("./userdict.txt")
    lcut = jieba.lcut(content, cut_all=True, HMM=True)
    print(lcut)


# 词性标注
def jieba5():
    # 词性标注就是在分词的基础上对每个词进行标注
    pesg_lcut = pesg.lcut("我爱北京天安门")
    # [pair("我", "r"), pair('爱', 'v'), pair('北京', 'ns'), pair('天安门', 'ns')]
    print(pesg_lcut)


if __name__ == '__main__':
    # jieba1()
    # jieba2()
    # jieba3()
    # jieba4()
    jieba5()