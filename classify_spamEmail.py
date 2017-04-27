#!usr/bin/env python
# -*-coding:utf-8 -*-
"""
使用朴素贝叶斯解决一些现实生活中的问题时,需要先从文本内容得到字符串列表,然后生成词向量。
bayes.py中的词向量是预先给定的, 现在从文本文档中构建自己的词列表。

示例: 过滤垃圾邮件
"""
import update_bayes as bayes
import numpy as np

'''准备数据:切分文本'''
# # 对于一个文本字符串,可以用Python的string.split()方法将其切分,但是标点符号也被当成词的一部分。
# mySent = 'I love machine learning.'
# pySplit = mySent.split()
# # print(pySplit)          # ['I', 'love', 'machine', 'learning.']
#
# # 可以用正则表达式来切分句子,其中分隔符是除单词、数字外的任意字符串。但是里面的空字符串需要去掉:
# # 计算每个字符串的长度,只返回长度大于0的字符串。
# import re
# regEx = re.compile('\\W*')
#
# listOfTokens = regEx.split(mySent)
# # print(listOfTokens)     # ['I', 'love', 'machine', 'learning', '']
# regSplit = [tok for tok in listOfTokens if len(tok) > 0]
# # print(regSplit)         # ['I', 'love', 'machine', 'learning']
# # 句子的第一个单词是大写的,如果目的是句子查找,那么这个特点会很有用。但这里的文本只看成词袋,所以希望所有词的形式是统一的。
# regSplitLower = [tok.lower() for tok in listOfTokens if len(tok) > 0]
# # print(regSplitLower)    # ['i', 'love', 'machine', 'learning']
#
# emailText = open('email/ham/6.txt').read()
# listOfTokens6 = regEx.split(emailText)
# regSplitLower6 = [tok.lower() for tok in listOfTokens6 if len(tok) > 0]
# print(regSplitLower6)


'''测试算法:使用朴素贝叶斯进行交叉验证'''
# 将文本解析器集成到一个完整分类器中
# 文件解析及完整的垃圾邮件测试函数
def text_parse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spam_test():
    docList = [];  classList = [];  fullText = []

    # 导入文件夹spam与ham下的文本文件,并将它们解析为词列表
    for i in range(1, 26):
        wordList = text_parse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)   # docList以每个txt为单位
        fullText.extend(wordList)  # fullText将每个txt合为一个
        classList.append(1)        # 垃圾邮件 类标签列表加1
        wordList = text_parse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = bayes.create_vocabList(docList)  # create vocabulary
    trainingSet = range(50);  testSet = []       # create test set

    # 随机构建训练集  这种随机选择数据的一部分作为训练集,而剩余部分作为测试机的过程称为"留存交叉验证(hold-out cross validation)"
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat = [];  trainClasses = []
    # 用训练集来获得概率
    for docIndex in trainingSet:                # train the classifier (get probs) train_NB0
        trainMat.append(bayes.bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = bayes.train_NB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    # 用测试集来计算错误率
    for docIndex in testSet:            # classify the remaining items
        wordVector = bayes.bagOfWords2VecMN(vocabList, docList[docIndex])   # docList中的词 是否在词汇表中出现
        #  wordVector: 文档向量,向量的每一元素为1或0
        if bayes.classify_NB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error: %s" % docList[docIndex])
    print('the error rate is: %s' % (float(errorCount) / len(testSet)))
    # return vocabList,fullText

spam_test()
