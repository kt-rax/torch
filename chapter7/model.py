# -*- coding: utf-8 -*-

# 3.模型定义:包括编码器解码器
import torch
from logger import logger
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

# from process import cut
from process import MAX_LENGTH

use_cuda = torch.cuda.is_available()
## 编码器
class EncoderRNN(nn.Module):
    '''
    编码器定义
    '''
    def __init__(self,input_size,hidden_size,n_layers=1):
        '''
        初始化
        Parameters
        ----------
        input_size : 输入向量长度，这里是词汇表大小
        hidden_size : 隐含层大小
        n_layers : 叠加层数
        Returns
        -------
        '''
        super(EncoderRNN,self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(input_size,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size)
        
    def forward(self,input,hidden):
        '''
        前向计算过程

        Parameters
        ----------
        input : 输入
        hidden : 隐含层状态
        Returns：编码器输出，隐含层状态
        '''
        try:
            embedded = self.embedding(input).view(1,1,-1)
            output = embedded
            for i in range(self.n_layers):
                output,hidden = self.gru(output,hidden)
            return output,hidden
        except Exception as err:
            logger.error(err)
            
    def initHidden(self):
        '''
        隐含层状态初始化
        Returns：初始化过的隐含层状态
        '''
        result = Variable(torch.zeros(1,1,self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

## 解码器
class DecoderRNN(nn.Module):
    '''
    解码器定义
    '''
    
    def __init__(self,hidden_size,output_size,n_layers = 1):
        '''
        初始化过程

        Parameters
        ----------
        hidden_size : 隐含层大小
        output_size : 输出大小
        n_layersf : 叠加层数
        Returns
        -------
        '''
        super(DecoderRNN,self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
    
        self.embedding = nn.Embedding(output_size,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size)
        self.out = nn.Linear(hidden_size,output_size)
        self.softmax = nn.LogSoftmax()
        
    def forward(self,input,hidden):
        '''
        前向计算过程
        Parameters
        ----------
        input : 输入信息
        hidden : 隐含层状态
        Returns：解码器输出，隐含层状态
        '''
        try:
            output = self.embedding(input).view(1,1,-1)
            for i in range(self.n_layers):
                output = F.relu(output)
                output,hidden = self.gru(output,hidden)
            output = self.softmax(self.out(output[0]))
            return output,hidden
        except Exception as err:
            logger.error(err)

    def initHidden(self):
        '''
        隐含层状态初始化
        Returns：初始化过的隐含层状态
        '''
        result = Variable(torch.zeros(1,1,self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
        
## 带注意力机制的解码器
class AttnDecoderRNN(nn.Module):
    '''
    带注意力的解码器的定义
    '''
    def __init__(self,hidden_size,output_size,n_layers = 1,dropout_p = 0.1,max_length = MAX_LENGTH):
        '''
        带注意力的解码器初始化过程
        hidden_size : 隐含层大小
        output_size : 输出大小
        n_layers : 叠加层数
        dropout_p : dropout率定义
        max_length : 接受的最大句子长度
        Returns
        '''
        super(AttnDecoderRNN,self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        
        self.embedding = nn.Embedding(self.output_size,self.hidden_size)
        self.attn = nn.Linear(self.hidden_size*2,self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size*2,self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size,self.hidden_size)
        self.out = nn.Linear(self.hidden_size,self.output_size)
        
    def forward(self,input,hidden,encoder_output,encoder_outputs):
        '''
        前向计算过程
        input : 输入信息
        hidden : 隐含层状态
        encoder_output : 编码器分时刻的输出
        encoder_outputs : 编码器全部输出
        Returns：解码器输出，隐含层状态，注意力权重
        '''
        try:
            embedded = self.embedding(input).view(1,1,-1)
            embedded = self.dropout(embedded)
            
            attn_weights = F.softmax(self.attn(torch.cat((embedded[0],hidden[0]),1)))
            attn_applied = torch.bmm(attn_weights.unsqueeze(0),encoder_outputs.unsqueeze(0))
            output = torch.cat((embedded[0],attn_applied[0]),1)
            output = self.attn_combine(output).unsqueeze(0)
            for i in range(self.n_layers):
                output = F.relu(output)
                output,hidden = self.gru(output,hidden)
            
            output = F.log_softmax(self.out(output[0]))
            
            return output,hidden,attn_weights
        except Exception as err:
            logger.error(err)
    
    def initHidden(self):
        '''
        隐含层状态初始化
        Returns：初始化过的隐含层状态
        '''
        result = Variable(torch.zeros(1,1,self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result