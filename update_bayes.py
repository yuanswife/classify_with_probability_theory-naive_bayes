#!usr/bin/env python
# -*-coding:utf-8 -*-
"""
朴素贝叶斯 朴素:是因为整个形式化过程只做最原始、最简单的假设
朴素贝叶斯的两个假设: 特征之间相互独立、每个特征同等重要
"""
import numpy as np

'''使用python进行文本分类:  将文本转换为数字向量、基于这些向量来计算条件概率、在此基础上构建分类器'''
# 准备数据:从文本中构建词向量
# 词表到向量的转换函数:
def load_dataset():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]       # 1 is abusive, 0 not
    return postingList, classVec


# 创建一个包含在所有文档中出现的不重复词的列表:
def create_vocabList(dataset):
    vocabSet = set([])                         # 创建一个空集
    for document in dataset:
        vocabSet = vocabSet | set(document)    # 创建两个集合的并集  注意:'|'只能用于set
    return list(vocabSet)


# 词集模型
# 获得词汇表后,便可以使用函数setOfWords2Vec()
def setOfWords2Vec(vocabList, inputSet):    # 输入: 词汇表、某个文档
    returnVec = [0] * len(vocabList)        # 创建一个和词汇表等长的向量,并将其元素都设置为0
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec                        # 输出:文档向量,向量的每一元素为1或0,分别表示词汇表中的单词在输入文档中是否出现。

# 准备数据:文档词袋模型  在词袋中每个单词可以出现多次,而在词集中,每个词只能出现一次。与setOfWords2Vec()唯一不同的是
# 每当遇到一个单词时,它会增加词向量中的对应值,而不只是将对应的数值设为1.
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


"""
p(ci|w) = p(w|ci)p(ci) / p(w)
p(ci) = 通过类别i中文档数 / 总的文档数
p(w|ci) 计算这个概率就要用到朴素贝叶斯假设。
        如果将w展开为一个个独立特征,那么就可以写作: p(w0,w1,w2..wN|ci)
        假设所有词都相互独立,该假设也称作条件独立性假设,意味着可以用  p(w0|ci)p(w1|ci)p(w2|ci)...p(wN|ci) 来计算

  利用贝叶斯分类器对文档进行分类时,要计算多个概率的乘积以获得文档属于某个类的概率,即计算p(w0|1)p(w1|1)p(w2|1).
  如果其中一个概率值为0,那么最后的乘积也为0.为降低这种影响,可以将所有词的出现数初始化为1,并将分母初始化为2.

  另一个问题是下溢出,这是因为太多很小的数相乘。当计算p(w0|ci)p(w1|ci)p(w2|ci)...p(wN|ci)时,由于大部分因子都非常小,所以
  程序会下溢出或者得到不正确的答案。一种解决办法是对乘积取自然对数
"""


# 朴素贝叶斯分类器训练函数
def train_NB0(trainMat, trainCategory):   # 输入:文档矩阵trainMatrix、由每篇文档类别标签所构成的向量trainCategory
    numTrainDocs = len(trainMat)
    numWords = len(trainMat[0])

    '''计算文档属于侮辱性文档(class=1)的概率,即 p(c1)'''
    pAbusive = np.sum(trainCategory) / float(numTrainDocs)

    '''计算p(wi|c1)  p(wi|c0)'''
    # 初始化程序中的分子(Numerator)变量和分母(Denominator)变量
    p0Num = np.ones(numWords);  p1Num = np.ones(numWords)
    p0Denom = 2.0;  p1Denom = 2.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:    # 该文档为侮辱性文档(class=1)
            # 向量相加
            p1Num += trainMat[i]
            p1Denom += np.sum(trainMat[i])  # 每条侮辱性文档的词数总和
        else:
            p0Num += trainMat[i]
            p0Denom += np.sum(trainMat[i])
    # 对每个元素除以该类别中的总词数  numpy可以很好实现,用一个数组除以浮点数即可
    p1Vec = np.log(p1Num / p1Denom)   # 侮辱性文档 每个词出现的概率
    p0Vec = np.log(p0Num / p0Denom)
    return p0Vec, p1Vec, pAbusive


# 朴素贝叶斯分类函数
def classify_NB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 将词汇表中所有词的对应值相加,然后将该值加到类别的对数概率上
    p1 = np.sum(vec2Classify * p1Vec) + np.log(pClass1)        # 元素相乘
    p0 = np.sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testing_NB():
    listOPosts, listClasses = load_dataset()
    myVocabList = create_vocabList(listOPosts)

    trainMat = [setOfWords2Vec(myVocabList, postinDoc) for postinDoc in listOPosts]  # (6, 32)
    p0V, p1V, pAb = train_NB0(np.array(trainMat), np.array(listClasses))

    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print('%s classified as: %s' % (testEntry, classify_NB(thisDoc, p0V, p1V, pAb)))

    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print('%s classified as: %s' % (testEntry, classify_NB(thisDoc, p0V, p1V, pAb)))


if __name__ == '__main__':
    listOPosts, listClasses = load_dataset()
    # print(listOPosts)
    myVocabList = create_vocabList(listOPosts)     # 包含在所有文档中出现的不重复词的列表
    # print(myVocabList)
    # ['cute', 'love', 'help', 'garbage', 'quit', 'I', 'problems', 'is', 'park', 'stop', 'flea', 'dalmation',
    # 'licks', 'food', 'not', 'him', 'buying', 'posting', 'has', 'worthless', 'ate', 'to', 'maybe', 'please',
    # 'dog', 'how', 'stupid', 'so', 'take', 'mr', 'steak', 'my']
    # print(len(myVocabList))    # 32
    vocabVec = setOfWords2Vec(myVocabList, listOPosts[0])  # 词汇表中的单词在输入文档中是否出现
    # print(vocabVec)
    # [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]
    # print(len(vocabVec))       # 32
    # 该for循环使用词向量来填充trainMat列表。
    trainMat = [setOfWords2Vec(myVocabList, postinDoc) for postinDoc in listOPosts]   # (6, 32)
    print(trainMat)

    # 属于侮辱性文档的概率pAb, 以及两个类别的概率向量p0V, p1V
    p0V, p1V, pAb = train_NB0(trainMat, listClasses)
    print(pAb)   # 0.5
    print(p0V)
    print(p1V)
    # [- 2.56494936 - 2.56494936 - 2.56494936 - 3.25809654 - 3.25809654 - 2.56494936
    #  - 2.56494936 - 2.56494936 - 3.25809654 - 2.56494936 - 2.56494936 - 2.56494936
    #  - 2.56494936 - 3.25809654 - 3.25809654 - 2.15948425 - 3.25809654 - 3.25809654
    #  - 2.56494936 - 3.25809654 - 2.56494936 - 2.56494936 - 3.25809654 - 2.56494936
    #  - 2.56494936 - 2.56494936 - 3.25809654 - 2.56494936 - 3.25809654 - 2.56494936
    #  - 2.56494936 - 1.87180218]

    # [- 3.04452244 - 3.04452244 - 3.04452244 - 2.35137526 - 2.35137526 - 3.04452244
    #  - 3.04452244 - 3.04452244 - 2.35137526 - 2.35137526 - 3.04452244 - 3.04452244
    #  - 3.04452244 - 2.35137526 - 2.35137526 - 2.35137526 - 2.35137526 - 2.35137526
    #  - 3.04452244 - 1.94591015 - 3.04452244 - 2.35137526 - 2.35137526 - 3.04452244
    #  - 1.94591015 - 3.04452244 - 1.65822808 - 3.04452244 - 2.35137526 - 3.04452244
    #  - 3.04452244 - 3.04452244]
    testing_NB()
    # ['love', 'my', 'dalmation'] classified as: 0
    # ['stupid', 'garbage'] classified as: 1
