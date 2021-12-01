# -*- coding: utf-8 -*-

# 2.数据处理:文件加载数据，数据解析，辅助函数
from __future__ import unicode_literals,print_function,division
import math
import re
import time
import unicodedata
import jieba

import torch
from logger import logger
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 25


def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c)!='Mn')

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])",r"\1",s)
    #
    # s = re.sub(r"[^a-zA-Z.!?]+",r" ",s)
    return s 


# 语言建模:对源语言和目标语言建模，保存语言相关的所有词的信息，可以在后续的训练与评估中使用 
class Lang:
    def __init__(self,name):
        '''
        添加 need_cut 可根据语种进行不同的分词逻辑处理 
        Parameters
        ----------
        name : 语种名称
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        self.name = name
        self.need_cut = self.name == 'cmn'
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:"SOS",1:"EOS"}
        self.n_words = 2 #初始化词数为2：SOS & EOS
        
    def addSentence(self,sentence):
        '''
        从语料中添加句子到Lang

        Parameters
        ----------
        sentence : TYPE语料中的每个句子
        '''
        if self.need_cut:
            sentence = cut(sentence)
        for word in sentence.split(' '):
            if len(word) > 0:
                self.addWord(word)
    
    def addWord(self,word):
        '''
        向Lang中添加每个词，并统计词频，如果是新词修改词表大小
        '''
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def cut(sentence,use_jieba=False):
    '''
    句子分词。
    Parameters
    ----------
    sentence : 要分词的句子
    use_jieba : 是否使用jieba进行智能分词，默认按单字切分
    Returns：分词结构，空格区分
    -------
    None.
    '''
    if use_jieba:
        return ' '.join(jieba.cut(sentence))
    else:
        words = [word for word in sentence]
        return ' '.join(words)

import jieba.posseg as pseq

def tag(sentence):
    words = pseq.cut(sentence)
    result = ''
    for w in words:
        result = result + w.word +'/'+w.flag+' '
    return result
# 数据处理：对语料中的句子进行处理，结果保存到各个语言的实例中
def readLangs(lang1,lang2,reverse = False):
    '''
    Parameters
    ----------
    lang1 : 源语言
    lang2 : 目标语言
    reverse : 是否进行逆向翻译
    Returns：源语言实例，目标语言实例，词语对
    '''
    # 读取txt文件并分割成行
    lines = open('data/%s-%s.txt'%(lang1,lang2),encoding='utf-8').read().strip().split('\n')
    
    # 按行处理成源语言-目标语言对，并做处理
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    
    # 生成语言实例
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    
    return input_lang,output_lang,pairs

eng_prefixes = ('I am','i m','he is','he s','she is','she s','you are','you re','we are','we re','they are','they re')

def filterPair(p):
    '''
    按自定义最大长度过滤

    '''
    #return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and p[1].startswith(eng_prefixes)
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH and \
           p[1].startswith(eng_prefixes)
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]   
def prepareData(lang1,lang2,reverse = False):
    input_lang,output_lang,pairs = readLangs(lang1,lang2,reverse)
    logger.info('Read %s sentence pairs' %len(pairs))
    pairs = filterPairs(pairs)
    logger.info('Trimmed to %s sentence pairs' %len(pairs))
    logger.info('Counting words ...')
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    logger.info('Counted words:')
    logger.info('%s,%d'%(input_lang.name,input_lang.n_words))
    logger.info('%s,%d'%(output_lang.name,output_lang.n_words))
    return input_lang,output_lang,pairs

def indexesFromSentence(lang,sentence):
    return [lang.word2index[word] for word in sentence.split(' ') if len(word)>0]

# 将指定的句子转换成Variable
def variableFromSentence(lang,sentence):
    if lang.need_cut:
        sentence = cut(sentence)
    # logger.info('cuted sentence:%s'%sentence)
    indexes = indexesFromSentence(lang,sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1,1))
    if use_cuda:
        return result.cuda()
    else:
        return result

# 指定的pair转换成Variable
def variablesFromPair(input_lang,output_lang,pair):
    input_variable = variableFromSentence(input_lang,pair[0])
    target_variable = variableFromSentence(output_lang,pair[1])
    return (input_variable,target_variable)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m*60
    return '%dm %ds' %(m,s)

def timeSince(since,percent):
    now = time.time()
    s = now - since 
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' %(asMinutes(s),asMinutes(rs))

if __name__ == '__main__':
    s = 'Fans of Belgium cheer prior to the 2018 FIFA World Cup Group G match between Belgium and Tunisia in Moscow, Russia, June 23, 2018.'
    s = '结婚的和尚未结婚的和尚'
    s = "买张下周三去南海的飞机票，海航的"
    s = "过几天天天天气不好。"
    
    a = cut(s,use_jieba=True)
    print(a)
    print(tag(a))
    
    
    
    
    
    
    
    
    
    
    
    