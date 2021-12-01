# -*- coding: utf-8 -*-
'''
## PCA
from sklearn import datasets
import torch
import numpy as np
import matplotlib.pyplot as plt

def PCA(data,k=2):
    #preprocess the data
    x = torch.from_numpy(data)
    x_mean = torch.mean(x,0)
    x = x - x_mean.expand_as(x)
    
    #svd
    U,S,V = torch.svd(torch.t(x))
    
    return torch.mm(x,U[:,:k])

iris = datasets.load_iris()
x = iris['data']
y = iris['target']
x_pca = PCA(x)
pca = x_pca.numpy()

plt.figure()
color = ['red','green','blue']
for i,target_name in enumerate(iris.target_names):
    plt.scatter(pca[y == i,0],pca[y == i,1],label = target_name,color = color[i])
plt.legend()
plt.title('PCA of iris dataset')
plt.show()

## AE
import os
import pdb
import torchvision
from torch import  nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torchvision import datasets
import numpy as np

# 配置参数
torch.manual_seed(1)
batch_size = 128 
learning_rate = 1e-2
num_epochs = 1

# 下载数据与预处理
train_dataset = datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)
test_dataset = datasets.MNIST(root='./data',train=False,transform=transforms.ToTensor(),download=True)

train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=10000,shuffle=False)

# AE模型
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28,1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000,500),
            nn.ReLU(True),
            nn.Linear(500,250),
            nn.ReLU(True),
            nn.Linear(250,2)
            )
        
        self.decoder = nn.Sequential(
            nn.Linear(2,250),
            nn.ReLU(True),
            nn.Linear(250,500),
            nn.ReLU(True),
            nn.Linear(500,1000),
            nn.ReLU(True),
            nn.Linear(1000,28*28),
            nn.Tanh()
            )
    
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = AutoEncoder()
# model = AutoEncoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-5)

# 模型训练
for epoch in  range(num_epochs):
    for data in train_loader:
        img,_ = data
        img = img.view(img.size(0),-1)
        img = Variable(img)
        
        output = model(img)
        loss = criterion(output,img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch [{} / {}],loss:{:.4f}'.format(epoch+1,num_epochs,loss.data.item()))

# 模型测试
model.eval()
eval_loss = 0
with torch.no_grad():
    for data in test_loader:
        img,label = data
        img = img.view(img.size(0),-1)
        img = Variable(img)
        
        label = Variable(label,volatile = True)
        out = model(img)
        y = (label.data).numpy()
        plt.scatter(out[:,0],out[:,1],c=y)
        plt.colorbar()
        plt.title('autocder of MNIST test dataset')
        plt.show()
'''
## AE 降噪
# 1.导库配置参数
import torch
import torch.nn as nn
import torch.utils as utils
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)
n_epochs = 200
batch_size = 100
learning_rate = 2*1e-4   
# 2.下载数据
mnist_train = dset.MNIST(root='./data',train=True,transform=transforms.ToTensor(),target_transform=None,download=True)
train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True)

# 3.模型定义
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,32,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2)
            )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,128,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,padding=1),
            nn.ReLU()         
            )
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(batch_size,-1)
        return out

encoder= Encoder()
# model = Encoder.cuda()
print(encoder)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(256,128,3,2,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128,128,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128,64,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64,64,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
            )
        
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64,32,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32,32,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32,1,3,2,1,1),
            nn.ReLU()
            )
    
    def forward(self,x):
        out = x.view(batch_size,256,7,7)
        out = self.layer1(out)
        out = self.layer2(out)
        
        return out

decoder = Decoder() # decoder = Decoder().cuda()
print(decoder)

# 4.训练
parameters = list(encoder.parameters()) + list(decoder.parameters())
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(parameters,lr=learning_rate)

noise = torch.rand(batch_size,1,28,28)
for I in range(n_epochs):
    for image,label in train_loader:
        image_n = torch.mul(image+0.25,0.1*noise)
        image = Variable(image)       #.cuda()
        image_n = Variable(image_n)   #.cuda()
        optimizer.zero_grad()
        output = encoder(image_n)
        print(output.shape)
        output = decoder(output)
        print(output.shape)
        loss = loss_func(output,image)
        loss.backward()
        optimizer.step()
        break
    print('epoch[{}/{}],loss:{:.4f}'.format(I+1,n_epochs,loss.data.item()))
    
# 5.测试
img = image[0].cpu()
input_img = image_n[0].cpu()
output_img = output[0].cpu()
origin = img.data.numpy()
inp = input_img.data.numpy()
out = output_img.data.numpy()
plt.figure('denoising autoencoder')
plt.subplot(131)
plt.imshow(origin[0],cmap='gray')
plt.subplot(132)
plt.imshow(inp[0],cmap='gray')
plt.subplot(133)
plt.imshow(out[0],cmap='gray')
plt.show()
print(label[0])












































