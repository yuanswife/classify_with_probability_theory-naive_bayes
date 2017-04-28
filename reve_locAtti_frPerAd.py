#!user/bin/env python
# -*- coding:utf-8 -*-
"""
示例: 使用朴素贝叶斯分类器从个人广告中获取区域倾向

收集数据: 从RSS源收集内容,这里需要对RSS源构建一个接口
准备数据: 将文本文件解析成词条向量
分析数据: 检查词条确保解析的正确性
训练算法: 使用我们之前建立的train_NB0()函数
测试算法: 观察错误率,确保分类器可用。可以修改切分程序,以降低错误率,提高分类结果
使用算法: 构建一个完整的程序,封装所有内容。给定两个RSS源,该程序会显示最常用的公共词
"""
# import classify_spamEmail as cs
import update_bayes as bayes
import numpy as np
import operator as op
import feedparser
# ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
# print(ny['entries'])
# print(len(ny['entries']))

def text_parse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


'''RSS源分类器及高频词去除函数'''
def calc_mostFreq(vocabList, fullText):
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)    # 计算词汇表中每个词在文本中出现频率
    sortedFreq = sorted(freqDict.iteritems(), key=op.itemgetter(1), reverse=True)
    return sortedFreq[:30]


# 与spam_test函数几乎相同,区别在于这里访问的是RSS源而不是文件。
def localWords(feed1, feed0):   # 使用两个RSS源作为参数。RSS源要在函数外导入,因为RSS源会随时间而改变。
    docList=[];  classList = [];  fullText =[]
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = text_parse(feed1['entries'][i]['summary'])   # 每次访问一条RSS源
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)           # NY is class 1

        wordList = text_parse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = bayes.create_vocabList(docList)       # create vocabulary
    top30Words = calc_mostFreq(vocabList, fullText)   # remove top 30 words   在词汇表中去掉出现次数最高的那些词
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])

    trainingSet = range(2*minLen); testSet=[]        # create test set
    for i in range(20):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat=[];  trainClasses = []
    for docIndex in trainingSet:    # train the classifier (get probs) trainNB0
        trainMat.append(bayes.bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = bayes.train_NB0(np.array(trainMat), np.array(trainClasses))

    errorCount = 0
    for docIndex in testSet:        # classify the remaining items
        wordVector = bayes.bagOfWords2VecMN(vocabList, docList[docIndex])
        if bayes.classify_NB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: %s' % (float(errorCount)/len(testSet)))
    return vocabList, p0V, p1V


# 最具表征性的词汇显示函数
def getTopWords(ny, sf):
    vocabList, p0V, p1V = localWords(ny, sf)

    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0:  topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:  topNY.append((vocabList[i], p1V[i]))

    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])

    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])


if __name__ == '__main__':
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    # vocabList, pSF, pNY = localWords(ny, sf)
    # print(vocabList, pSF, pNY)
    getTopWords(ny, sf)
