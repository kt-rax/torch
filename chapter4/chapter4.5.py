# -*- coding: utf-8 -*-

# 1.配置库与参数
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# 超参
torch.manual_seed(1)
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# 2.加载mnist数据
train_dataset = dsets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)
test_dataset = dsets.MNIST(root='./data',train=False,transform=transforms.ToTensor(),download=True)

# 3.数据批处理
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size=batch_size,shuffle=False)

# 4.创建DNN模型
class Net(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    def forward(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

net = Net(input_size,hidden_size,num_classes)
print(net)
# 5.训练流程
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr = learning_rate)

for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        # convert torch tensor to Varaiable
        images = Variable(images.view(-1,28*28))
        labels = Variable(labels)
        # forward + backward + optimize
        optimizer.zero_grad    #梯度清零，以免影响其他batch 
        outputs = net(images)  #前向传播  
        loss = criterion(outputs,labels)
        loss.backward()    #后向传播，梯度计算 
        optimizer.step()   # 梯度更新
        
        if(i+1)%100 == 0:
            print('Epoch[%d/%d],Step[%d/%d],Loss:%.4f'%(epoch+1,num_epochs,i+1,len(train_dataset)//batch_size,loss.data.item()))

# 6.在测试集测试识别率
correct = 0
total = 0
for images,labels in test_loader:
    images = Variable(images.view(-1,28*28))
    outputs = net(images)
    _,predicted = torch.max(outputs.data,1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images :%d %%'%(100*correct/total))
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
