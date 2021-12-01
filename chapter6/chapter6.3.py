# -*- coding: utf-8 -*-

## word2vec

# 1.加库配参
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
N_EPOCHS = 300

# 2.数据准备

test_sentence = """From what we learned above we can already tell that this additional cost will cause the resulting weights to be penalized. Unfortunately L1 regularization does not have a closed form solution because it is not differentiable when a weight β falls to 0. Thus this requires some more work to solve. LASSO is an algorithm for finding the solution.

First let's remember that our constraint here is ||β||≤c where c is a real number which is inversely related to λ. The plan is to find an appropriate λ such that the minimum of the function falls within or on the contour of the constraint. This constraint has very sharp edges which lie on each dimensional axis at a distance c from the origin. You can imagine it like a diamond in 2D space, an octohedron in 3D space, etc.

In high dimensional space these spikes have a very high likelyhood to being hit by the function you wish to optimize thus this will cause many of the features to have an associated weight of 0. For example if we compare the regularization of this line in 2D space using both L2 and L1 regularization.""".split()
'''
test_sentence = """Word embeddings are dense vectors of real numbers, 
one per word in your vocabulary. In NLP, it is almost always the case 
that your features are words! But how should you represent a word in a
computer? You could store its ascii character representation, but that
only tells you what the word is, it doesn’t say much about what it means 
(you might be able to derive its part of speech from its affixes, or properties 
from its capitalization, but not much). Even more, in what sense could you combine 
these representations?""".split()
'''

trigrams = [([test_sentence[i],test_sentence[i+1]],test_sentence[i+2])
            for i in range(len(test_sentence)-2)]
vocab = set(test_sentence)
word_to_ix = {word: i for i,word in enumerate(vocab)}
idx_to_word = {word_to_ix[word]:word for word in word_to_ix}

# 3.语言模型
class NGramLanguageModeler(nn.Module):
    def __init__(self,vocab_size,embedd_dim,context_size):
        super(NGramLanguageModeler,self).__init__()
        self.embeddings = nn.Embedding(vocab_size,embedd_dim)
        self.linear1 = nn.Linear(context_size*embedd_dim,128)
        self.linear2 = nn.Linear(128,vocab_size)
        
    def forward(self,inputs):
        embeds = self.embeddings(inputs).view(1,-1)
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_prob = F.log_softmax(out)
        return log_prob

# 4.loss与优化器
losses = []
loss_func = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab),EMBEDDING_DIM,CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(),lr = 0.001)
        
# 5.训练
for epoch in range(N_EPOCHS):
    total_loss = torch.Tensor([0])
    for context,target in trigrams:
        # step1.准备数据
        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))
        # step2.梯度初始化
        model.zero_grad()        
        # step3.前向算法
        log_prob = model(context_var)        
        # step4.loss
        loss = loss_func(log_prob,autograd.Variable(torch.LongTensor([word_to_ix[target]])))        
        # step5.后向算法与梯度更新
        loss.backward()
        optimizer.step()
        # step6.loss
        total_loss += loss.data
    print('\r epoch[{}] - loss:  {:.6f}'.format(epoch,total_loss[0]))      

# 6.测试
word ,label = trigrams[3]
word = autograd.Variable(torch.LongTensor([word_to_ix[i] for i in word]))
out = model(word)
_,predict_label = torch.max(out,1)
predict_word = idx_to_word[predict_label.data[0].item()]
print('real word is {},predict word is  {}'.format(label,predict_word))























