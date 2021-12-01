# -*- coding: utf-8 -*-

### 基于GRU和Attention的机器翻译

# 5.训练和模型保存 
        
# 引库
import pickle
import sys
from io import open

import torch
from logger import logger
from model import AttnDecoderRNN
from model import  EncoderRNN
from model import  DecoderRNN
from process import prepareData
from train import  *
from torch.utils.tensorboard import SummaryWriter

use_cuda = torch.cuda.is_available()
logger.info('Use cuda:{}'.format(use_cuda))
input = 'eng'
#output = 'fra'

output = 'cmn'
# 从参数接收要翻译的语种名词
if len(sys.argv) > 1:
    output = sys.argv[1]
logger.info('%s -> %s'%(input,output))

# 语料处理
input_lang,output_lang,pairs = prepareData(input,output,True)
logger.info(random.choice(pairs))

# 查看两种语言的词汇大小情况
logger.info('input_lang.n_words: %d'%input_lang.n_words)
logger.info('output_lang.n_words: %d' %output_lang.n_words)

# 保存处理过的语言信息，评估时加载使用
pickle.dump(input_lang,open('./data/%s_%s_input_lang.pkl' %(input,output),'wb'))
pickle.dump(output_lang,open('./data/%s_%s_output_lang.pkl'%(input,output),'wb'))
pickle.dump(pairs,open('./data/%s_%s_pairs.pkl'%(input,output),'wb'))
logger.info('lang-saved.')

# 训练
#解码器和编码器的实例化
hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words,hidden_size)
attn_decoder1 = AttnDecoderRNN(hidden_size,output_lang.n_words,1,dropout_p = 0.1)


writer = SummaryWriter()
### plot model

dummy_encoder_input_1 = torch.randn((1)).long()
dummy_encoder_input_2 = torch.randn((1, 1, 256)).float()
dummy_decoder_input_1 = torch.randn((1, 1)).long()
dummy_decoder_input_2 = torch.randn((1,1,256)).float()
dummy_decoder_input_3 = torch.randn((1,1,256)).long()
dummy_decoder_input_4 = torch.randn((25,256)).float()

encoder_output, encoder_hidden = encoder1(dummy_encoder_input_1,dummy_encoder_input_2)
decoder_output, decoder_hidden, decoder_attention = attn_decoder1(dummy_decoder_input_1,dummy_decoder_input_2,dummy_decoder_input_3,dummy_decoder_input_4)

decoder1 = DecoderRNN(hidden_size, output_lang.n_words,3)
decoder1_output, decoder1_hidden = decoder1(dummy_decoder_input_1,dummy_decoder_input_2)

torch.onnx.export(encoder1,args=(dummy_encoder_input_1,dummy_encoder_input_2), f='encoder1.onnx', input_names=['input_variable','eoncoder_hidden'], output_names=['encoder_output','encoder_hidden'])
torch.onnx.export(attn_decoder1,args=(dummy_decoder_input_1,dummy_decoder_input_2,dummy_decoder_input_3,dummy_decoder_input_4), f='attn_decoder1.onnx', 
                  input_names=['decoder_input','decoder_hidden','encoder_output','encoder_ouputs'], output_names=['decoder_outputs','decoder_hidden','decoder_attention'])
torch.onnx.export(decoder1,args=(dummy_decoder_input_1,dummy_decoder_input_2), f='decoder1.onnx', 
                  input_names=['decoder_input','decoder_hidden'], output_names=['decoder_outputs','decoder_hidden'])

writer.add_graph(encoder1,input_to_model=(dummy_encoder_input_1,dummy_encoder_input_2))
writer.close()

if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()

logger.info('train start. ')
trainIters(input_lang,output_lang,pairs,encoder1,attn_decoder1,75000,print_every=5000)
logger.info('train end. ')

# 保存模型
# 保存编码器和解码器的网络状态
torch.save(encoder1.state_dict(),open('./data/%s_%s_encoder1.stat' %(input,output),'wb'))

torch.save(attn_decoder1.state_dict(),open('./data/%s_%s_attn_decoder1.stat'%(input,output),'wb'))
logger.info('stat saved. ')

# 保存整个网络
torch.save(encoder1,open('./data/%s_%s_encoder1.model'%(input,output),'wb'))
torch.save(attn_decoder1,open('./data/%s_%s_attn_decoder1.model'%(input,output),'wb'))
logger.info('model saved. ')




































