# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.models.vgg import model_urls
from torchviz import make_dot

batch_size = 3
learning_rate =0.0002
epoch = 50

resnet = models.resnet50(pretrained=True)
print(resnet)
make_dot(resnet)
