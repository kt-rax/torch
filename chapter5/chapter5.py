# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

input = torch.autograd.Variable(torch.randn(20,16,50,100))
# 方形卷积核和等长的步长 
m = nn.Conv2d(16,33,3,stride=2)
output = m(input)
print(input.shape)
print(output.shape)

# 非方形的卷积核，非等长的步长和边界填充
m = nn.Conv2d(16,33,(3,5),stride =(2,1),padding=(4,2))
output = m(input)
#print(input.shape)
print(output.shape)

# 非方形卷积核，非等长的步长，边界填充和空间间隔
m = nn.Conv2d(16,33,(3,5),stride=(2,1),padding=(4,2),dilation=(3,1))
output = m(input)
#print(input.shape)
print(output.shape)


# 池化层
from torch.autograd import Variable
input = Variable(torch.randn(20,16,50,32))

m = nn.MaxPool2d(2,stride = 2)
output = m(input)
print(output.shape)

m = nn.MaxPool2d((3,2),stride=(2,1))
output = m(input)
print(output.shape)

## LeNet-5
import torch.nn.functional as F
import torch.nn as nn
from torchviz import make_dot

class LeNet(nn.Module):
    def __inti__(self):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.Conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        
    def foward(self,x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out,2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out,2)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out
#x = torch.zeros(50,3,224,224,dtype=torch.float,requires_grad=False)
#model_Le = LeNet(x)
#print(model_Le)
#make_dot(model_Le)       
## AlexNet
class AlexNet(nn.Module):
    def __init__(self,num_classes):
        super(AlexNet,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,96,kernel_size = 11,stride = 4,padding = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.Conv2d(64,256,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            )
        
        self.claasifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,num_classes),
            )
        
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),256*6*6)
        x = self.claasifier(x)
        
        return x

# model_Alex = AlexNet(10)
# print(model_Alex)
# make_dot(model_Alex) 
## VGGNet
cfg = {
       'VGG11':[64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
       'VGG13':[64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'],
       'VGG16':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
       'VGG19':[64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M'],
       }

class VGG(nn.Module):
    def __init__(self,vgg_name):
        super(VGG,self).__init__()
        self.feature = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512,10)
        
    def forward(self,x):
        out = self.features(x)
        out = out.view(out.size(0),-1)
        out = self.classifier(out)
        
        return out
    
    def _make_layers(self,cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
            else:
                layers += [nn.Conv2d(in_channels,x,kernel_size=3,padding =1),nn.BatchNorm2d(x),nn.ReLU(inplace=True)]
                in_channels = x
            layers += [nn.AvgPool2d(kernel_size=1,stride=1)]
            
            return nn.Sequential(*layers)


# model_vgg = VGG()
# print(model_vgg)
# make_dot(model_vgg)        
### mnist 数据集上卷积神经网络的实现
# 1.导库
import torch
from torch import nn,optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets    
from torchviz import make_dot  
import torch.onnx      
    
torch.manual_seed(1)
batch_size = 128
learning_rate = 1e-2
num_epochs = 1
        
# 2.加载数据
train_dataset = datasets.MNIST(root = './data',train = True, transform = transforms.ToTensor(),download = True)
test_dataset = datasets.MNIST( root = './data',train = False,transform = transforms.ToTensor(),download = True)
train_loader = DataLoader(train_dataset,batch_size = batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size = batch_size,shuffle=False)
    
# 3.创建cnn
class Cnn(nn.Module):
    def __init__(self,in_dim,n_class):
        super(Cnn,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, 3,stride = 1,padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            # nn.MaxPool2d(2,2)
            nn.Conv2d(6,16,5,stride = 1,padding = 0),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2,2)
            )
        
        self.fc = nn.Sequential(
            nn.Linear(400, 120),
            nn.Linear(120,84),
            nn.Linear(84,10),
            )
        
    def forward(self,x):
        out = self.conv(x)
        out = out.view(out.size(0),400)
        out = self.fc(out)
        
        return out
    
model = Cnn(1,10)
print(model)
#make_dot(model).render('mnist_cnn',format='png')
# 4.模型训练
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr = learning_rate)

for epoch in range(num_epochs):
    running_loss = 0.0
    running_acc = 0.0
    for i,data in enumerate(train_loader,1):
        img ,label = data
        img ,label = Variable(img),Variable(label)
        #前向传播
        out = model(img)
        loss = criterion(out, label)
        running_loss += loss.data.item() * label.size(0)
        _,pred = torch.max(out,1)
        num_correct = (pred == label).sum() 
        #accuracy = (pred == label).float().mean()
        running_acc += num_correct.data.item()      
        #梯度清零
        optimizer.zero_grad()
        #后向传播
        loss.backward()
        optimizer.step()
    print('Train {} epoch,Loss:{:.6f},Acc:{:.6f}'.format(epoch+1,running_loss/(len(train_dataset)),running_acc/(len(train_dataset))))
make_dot(out).render('mnist_cnn',format='png')


#torch.onnx.export(model,'rnn.onnx',f=out, input_names='input_names', output_names='output_names')     
            
# 5.测试集测试
model.eval()
eval_loss = 0.0
eval_acc = 0.0
for data in test_loader:
    img, label = data
    img, label = Variable(img), Variable(label)
    out = model(img)
    loss = criterion(out, label)
    eval_loss += loss.data.item()*label.size(0)
    _,pred = torch.max(out,1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.data.item()
    
print('Test Loss:{:.6f},Acc:{:.6f}'.format(eval_loss/(len(test_dataset)),eval_acc*1.0/(len(test_dataset))))

    
torch.onnx.export(model,img,input_names='input_names', output_names='output_names')     
          
 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        