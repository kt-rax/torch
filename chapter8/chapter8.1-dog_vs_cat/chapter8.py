# -*- coding: utf-8 -*-
## step2.配置库与参数
from __future__ import print_function,division
import shutil
import torch
import os
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,datasets,utils,models
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

# 配置参数
random_state = 1
torch.manual_seed(random_state )
torch.cuda.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)
np.random.seed(random_state)
epochs = 10
batch_size = 4
num_workers = 24

# step3.加载数据并做图像预处理：对加载图像做归一化处理，并裁剪为224*224大小
data_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

# 数据的批处理，大小尺寸为batch_size
# 在训练集中，shuffle必须设置为True，表示次序是随机的
train_dataset = datasets.ImageFolder(root='cats_and_dogs_small/train/',transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
test_dataset = datasets.ImageFolder(root='cats_and_dogs_small/test/',transform=data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)

# step4.创建神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.maxpool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*53*53,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,2)
    def forward(self,x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = x.view(-1,16*53*53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


net = models.resnet18(pretrained = True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs,2)

net = net.cuda()

#net = Net().cuda()
print(net)


## step5.整体训练与测试框架
use_gpu = torch.cuda.is_available()
optimizer = optim.SGD(net.parameters(),lr=0.0001,momentum=0.9)
criterion = nn.CrossEntropyLoss()

net.train()
for epoch in range(epochs):
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    for i,data in enumerate(train_loader,0):
        inputs,train_labels = data
        if use_gpu:
            inputs,labels = Variable(inputs.cuda()),Variable(train_labels.cuda())
        else:
            inputs,labels = Variable(inputs),Variable(train_labels)
        # inputs,labels  = Variable(inputs),Variable(train_labels)
        optimizer.zero_grad()
        outputs = net(inputs)
        _,train_predicted = torch.max(outputs.data,1)
        
        train_correct += (train_predicted == labels.data).sum()
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.data.item()
        train_total += train_labels.size(0)
    print('train %d epoch loss: %.3f   acc: %.3f '%(epoch+1,running_loss/train_total,100*train_correct/train_total))
    #模型测试
    correct = 0
    test_loss = 0.0
    test_total = 0
    net.eval()
    for data in test_loader:
        images,labels =data
        if use_gpu:
            images,labels = Variable(images.cuda()),Variable(labels.cuda())
        else:
            images,labels = Variable(images),Variable(labels)
        outputs = net(images)
        _,predicted = torch.max(outputs.data,1)
        loss = criterion(outputs,labels)
        test_loss += loss.data.item()
        test_total += labels.size(0)
        correct += (predicted == labels.data).sum()
    print('test %d epoch loss:%.3f acc: %.3f '%(epoch+1,test_loss/test_total,100*correct/test_total))





































