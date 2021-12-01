# -*- coding: utf-8 -*-

import os
# 特征提取，提取音频的频谱特征，需要调用librosa库来提取特征
AUDIO_EXTENSIONS = ['.wav','.WAV']

# 判断是否是音频文件
def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)

# 找到类名并索引
def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir,d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes,class_to_idx

# 构造数据集
def make_dataset(dir,class_to_idx):
    spects = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir,target)
        if not os.path.isdir(d):
            continue
        for root,_,fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_audio_file(fname):
                    path = os.path.join(root,fname)
                    item = (path,class_to_idx[target])
                    spects.append(item)
    return spects,item

# 频谱加载器，处理音频，生成频谱
def spect_loader(path,window_size,window_stride,window,normalize,max_len = 101):
    y,sr = librosa.load(path,sr=None)
    # n_fft = 4096
    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)
    
    # 短时傅里叶变换
    D = librosa.stft(y,n_fft=n_fft,hop_length=hop_length,win_length=win_length,window=window)
    spect,phase = librosa.magphase(D)  # 计算幅度谱和相位
    # S = log(S+1)
    spect = np.log1p(spcet)  # 计算log域f幅度谱
    # 处理所有的频谱，使得长度一致
    # 少于规定长度，补0到规定长度，多于规定长度的，截短到规定长度
    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0],max_len - spect.shape[1]))
        spect = np.hstack((spect,pad))
    else spect.shape[1] > max_len:
        spcet = spect[:max_len,]
    spect = np.resize(spect,(1,spect.shape[0],spect.shape[1]))
    spect = torch.FloatTensor(spect)
    
    # z-score归一化
    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)
    return spect

# 音频加载器，类似Pytorch的加载器，实现对数据的加载
class SpeechLoader(data.Dataset):
    '''Google 音频命令数据集的数据形式如下
    root/one/xxx.wav
    root/head/123.wav
    参数：
    root(string):原始数据集路径
    window_size:STFT的窗口大小，默认参数是：.02
    window_stride:用于STFT窗的帧移是.01
    window_type:窗的类型，默认是hamming窗
    normalize：布尔型变量，频谱是否进行归一化，归一化后频谱均值为零，方差为1
    max_len:帧的最大长度
    属性：
    classes(list):类别名的列表
    class_to_idx(dict):目标参数（class_name,class_index）(字典类型)
    spects(list):频谱参数（spects path,class_index）的列表
    STFT parameter:窗长，帧移，窗的类型，归一化   
    '''
    def __init__(self,root,window_size = .02,window_stride = .01,window_type = 'hamming',normalize = True,
                 max_len=101):
        classes,class_to_idx = find_classes(root)
        spects = make_dataset(root,class_to_idx)
        if len(spects) == 0: # 错误处理
            raise(RuntimeError('Found 0 sound files in subfolders of : '+root+'Supported audio file extentions are:'+','.join(AUDIO_EXTENSIONS)))
        self.root = root
        self.spects = spects
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.loader = spect_loader
        self.window_size = window_size
        self.window_stride = window_stride
        self.normalize = normalize
        self.max_len = max_len
    
    def __getitem__(self,index):
        '''
        Args:
            index(int):序列
        Returns：
            tuple（spect,target）:返回（spec，target）,其中target是类别名的索引
        '''
        path,target = self.spects[index]
        spect = self.loader(path,self.window_size,self.window_stride,self.window_type,self.normalize,self.max_len)
        return spect,target
    
    def __lenn__(self):
        return len(self.spects)


## 数据准备：加载数据：训练集，验证集，测试集

train_dataset = SpeechLoader(args.train_path,window_size=args.window_size,window_stride=args.window_stride,
                             window_type=args.window_type,normalize=args.normalize)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=20,
                                           pin_memory=args.cuda,sampler=None)
valid_dataset = SpeechLoader(args.valid_path,window_size=args.window_size,window_stride=args.window_stride,
                             window_type,args.window_type,normalize=args.normalize)
valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=args.batch_size,shuffle=None,num_workers=20,
                                           pin_memory=args.cuda,sampler=None)
test_dataset = SpeechLoader(args.test_path,window_size=args.window_size,window_stride=args.window_stride,
                            window_type=args.window_type,normalize=args.normalize)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.test_batch_size,shuffle=None,num_workers=20,
                                          pin_memory=args.cuda,sampler=None)

## 建立模型：利用卷积神经网络VGG建立命令词识别模型
def _make_layers(cfg):
    layers = []
    in_channels = 1
    for x in cfg:
        if x == 'M': # maxpool池化层
            layers += [nn.MaxPool2d(kernel_size=2,strides=2)]
        else:  # 卷积层
            layers += [nn.Conv2d(in_channels,x,kernel_size=3,padding=1),nn.BatchNorm2d(x),nn.ReLU(inplace=True)]
            in_channels = x
    layers += [nn.AvgPool2d(kernel_size=1,stride=1)] #avg池化
    return nn.Sequential(*layers)

# 各个VGG模型的参数
cfg =[
     'VGG11':[64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M']
     'VGG13':[64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M']
     'VGG16':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M']
     'VGG19':[64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M']]
        
# VGG卷积神经网络
class VGG(nn.Module):
    def __init__(self,vgg_name):
        super(VGG,self).__init__()
        self.features = _make_layers(cfg[vgg_name])  # VGG的模型层
        self.fc1 = nn.Linear(7680,512)
        self.fc2 = nn.Linear(512,30)
    
    def forward(self,x):
        out = self.features(x)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = self.fc2(out)
        return F.log_softmax(out) # log_softmax激活函数
    
## 训练算法和测试算法
# 训练函数，模型在train集上训练
def train(loader,model,optimizer,epoch,cuda):
    model.train()
    train_loss = 0
    train_correct = 0
    for batch_idx,(data,target) in enumerate(loader):
        if cuda:
            data,target = data.cuda(),target.cuda()
        data,target = Variable(data),Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.null_loss(output,target)
        loss.backward()
        optimizer.step()
        train_loss += loss.data.item()
        pred = output.data.max(1,keepdim=True)[1]
        train_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    train_loss = train_loss / len(loader.dataset)
    print('train set: Average loss : {:.4f}, Accuracy:{}/{}({:.0f}%)'.format(train_loss,train_correct,
                   len(loader.dataset),100.0*train_correct/len(loader.dataset)))

# 测试函数,用来测试valid集和test集
def test(loader,model,cuda):
    model.eval()
    test_loss = 0
    correct = 0
    for data,target in loader:
        if cuda:
            data,target = data.cuda(),target.cuda()
        data,target = Variable(data,volatile=True),Variable(target)
        output = model(data)
        test_loss += F.null_loss(output,target,size_average=False).data[0]
        pred = output.data.max(1,keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(loader.dataset)
    print(loader.split('_')[0]+' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.9f}%)'.format(test_loss,
                 correct,len(loader.dataset),100. * correct / len(loader.dataset)))
    

### 整体算法
# 参数设置
parser = argparse.ArgumentParser(description='Google Speech Commands Recognition')
parser.add_argument('-train_path',default='./data/train',help='path to the train data folder')
parser.add_argument('-test_path',default='./data/test',help='path to the test data folder')
parser.add_argument('-valid_path',default='./data/valid',help='path to the valid data folder')
parser.add_argument('-batch_size',type=int,default=100,metavar='N',help='training and valid batch size')
parser.add_argument('-test_batch_size',type=int,default=100,metavar='N',help='batch size for testing')
parser.add_argument('-arc',default='VGG11',help='network architecture:VGG11,VGG13,VGG16,VGG19')
parser.add_argument('-epochs',type=int,default=10,metavar='N',help='number of epochs to train')
parser.add_argument('-lr',type=float,default=0.001,metavar='LR',help='learning_rate')
parser.add_argument('-momentum',type=float,default=0.9,metavar='M',help='SGD momentum, for SGD only')
parser.add_argument('-optimizer',default='adam',help='optimization method:sgd | adam')
parser.add_argument('-cuda',default=True,help='enable CUDA')
parser.add_argument('-seed',type=int,default=1234,metavar='S',help='random seed')
# 特征提取参数设置
parser.add_argument('-window_size',default=.02,help='window size for the stft')
parser.add_argument('-window_stride',default=.01,help='window stride for the stft')
parser.add_argument('-window_type',default='hamming',help='window type for the stft')
parser.add_argument('-normalize',default=True,help='boolean,wheather or not to normalize the spect')
args = parser.parse_args()
# 确定是否使用CUDA
args.cuda = args.cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)  # Pytorch 随机种子设置
if args.cuda:
    torch.cuda.manual_seed(args.seed)  # CUDA随机种子设置

# 加载数据，训练集，验证集，测试集
train_dataset = SpeechLoader(args.train_path,window_size=args.window_size,window_stride=args.window_stride
                             window_type=args.window_type,normalize=args.normalize)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=20,
                                           pin_memory=args.cuda,sampler=None)
valid_dataset = SpeechLoader(args.valid_path,window_size=args.window_size,window_stride=args.window_stride,
                             window_type=args.window_type,normalize=args.normalize)
valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=args.batch_size,shuffle=None,num_workers=20,
                                           pin_memory=args.cuda,sampler=None)
test_dataset = SpeechLoader(args.test_path,window_size=args.window_size,window_stride=args.window_stride,
                            window_type=args.window_type,normalize=args.normalize)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.test_batch_size,shuffle=None,number_of_samples=20,
                                          pin_memory=args.cuda,sampler=None)
# 建立模型
model = VGG(args.arc)
if args.cuda:
    print('Using CUDA with {0} GPUS'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model).cuda()
    
# 定义优化器
if args.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model.parameters(),lr = args.lr)
elif args.optimizer.lower() == 'sgd':
    optimizer = optim.SGD(model.parameters(),lr = args.lr,moments=args.momentum)

# train 和 valid 过程

for epoch in range(1,args.epochs+1):
    # 模型在train 集上训练
    train(train_loader,model,optimizer,epoch,args.cuda)
    # 验证集测试
    test(valid_loader,model,args.cuda)
    # 测试集验证
    test(test_loader,model,args.cuda)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    













