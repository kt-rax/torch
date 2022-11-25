# -*- coding: utf-8 -*-

## 数据准备和Torchtext
import os
import torch
import urllib
import tarfile
from torchtext import Dataset
import re
import torch.nn as nn
import torch.nn.functional as F

class TarDataset(Dataset):
    '''
    定义一个数据集，该数据集从一个可下载的tar文件地址下载
    url:URL地址，指向一个可下载的tar文件地址
    filename:文件名，该可下载的tar文件的文件名
    dirname:文件夹名字，上层的文件夹名字，该文件夹下包含数据的压缩文件
    '''
    @classmethod
    def download_or_unzip(cls,root):
        path = os.path.join(root,cls.dirname)
        if not os.path.isdir(path):
            tpath = os.path.join(root,cls.filename)
            if not os.path.isfile(tpath):
                print('downloading')
                urllib.request.urlretrieve(cls.url,tpath)
            with tarfile.open(tpath,'r') as tfile:
                print('extracting')
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tfile, root)
        return os.path.join(path,'')

# 调用torchtext对20Newsgroups数据集进行处理，由于20Newsgroups数据集不在torchtext函数，需要从头构建类20Newsgroups
class NEWS_20(TarDataset):
    url = 'http://people.csail.mit.edu/jrennie/20Newsgroups/20news-bydata.tar.gz'
    filename = 'data/20news-bydate-train'

    dirname = ''
    @staticmethod
    def sort_key(ex):
        return len(ex.text)
    
    def __init__(self,text_field,label_field,path=None,text_cnt=1000,examples=None,**kwargs):
        '''
        根据路径和域创建一个MR数据集实例
        Parameters
        ----------
        text_field : 该域用于文本数据
        label_field : 该域用于标注数据
        path : 数据文件的路径
        text_cnt :包含所有数据的实例
        examples : TYPE, optional
            DESCRIPTION. The default is None.
        **kwargs : TYPE
        '''
        def clean_str(string):
            '''
            所有数据集的分词、字符清理，不包括SST，代码从
            https://github.com/yoonkim/CNN_sentence/lib/master/process_data.py 取得
            '''
            string = re.sub(r"[^A-Za-z0-9(),!?\'\']","",string)
            string = re.sub(r"\'s","\'s",string)
            string = re.sub(r"\'ve","\'ve",string)
            string = re.sub(r"n\'t","n\'t",string)
            string = re.sub(r"\'re","\'re",string)
            string = re.sub(r"\'d","\'d",string)
            string = re.sub(r"\'ll","\'ll",string)
            string = re.sub(r",",",",string)
            string = re.sub(r"!","!",string)
            string = re.sub(r"\)","\)",string)
            string = re.sub(r"\?","\?",string)
            string = re.sub(r"\s{2,}","",string)
            
            return string.strip().lower()
        
        text_field.processing = data.Pipeline(clean_str)
        fields = [('text',text_field),('label',label_field)]
        categories = ['alt.atheism','comp.graphics','sci.med','soc.religion.christian']
        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            for sub_path in categories:
                sub_path_one = os.path.join(path,sub_path)
                sub_path_two = os.listdir(sub_path_one)
                cnt = 0
                for sub_path_two in sub_path_two:
                    lines = ""
                    with open(os.path.join(sub_path_one,sub_path_two),encoding='utf8',errors='ignore') as f:
                        lines = f.read()
                    examples += [data.Example.fromlist([lines,sub_path],fields)]
                    cnt += 1
        super(NEWS_20,self).__init__(examples,fields,**kwargs)
    
    @classmethod
    def splits(cls,text_field,label_field,root='./data',train='20news-bydata-train',test='20news-bydata-test',**kwargs):
        '''
        
        '''
        path = cls.download_or_unzip(root)
        train_data = None if train is None else cls(
            text_field,label_field,os.path.join(path,train),2000,**kwargs)
        dev_ratio = 0.1
        dev_index = -1*int(dev_ratio*len(train_data))
        return (cls(text_field,label_field,examples=train_data[:dev_index]),
                cls(text_field,label_field,examples=train_data[dev_index:]))