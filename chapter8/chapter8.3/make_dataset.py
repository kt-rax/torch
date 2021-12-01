# -*- coding: utf-8 -*-
### 命令词识别的pytorch实现

# 数据集划分 speech_commands_v0.01.tar.gz

import os
import  shutil
import argparser
#把文件从源文件夹移动到目标文件夹 make_dateset.py
def move_files(original_fold,data_fold,data_filename):
    with open(data_filename) as f:
        for line in f.readlines():
            vals = line.split('/')
            dest_fold = os.path.join(data_fold,vals[0])
            if not os.path.exits(dest_fold):
                os.mkdir(dest_fold)
            shutil.mv(os.path.join(original_fold,line[:-1]),os.path.join(data_fold,line[:-1]))

# 建立 train文件夹
def create_train_fold(original_fold,train_fold,test_fold):
    # 文件夹名列表
    dir_names = list()
    for file in os.listdir(test_fold):
        if os.path.isdir(os.path.join(test_fold,file)):
            dir_names.append(file)
    # 建立训练文件夹train
    for file in os.listdir(original_fold):
        if os.path.isdir(os.path.join(test_fold,file)) and file in dir_names:
            shutil.mv(os.path.join(original_fold,file), os.path.join(train_fold,file))

# 建立数据集 train,valid,test
def make_dataset(gcommands_fold,out_path):
    validation_path = os.path.join(gcommands_fold,'validation_list.txt')
    test_path = os.path.join(gcommands_fold,'test_list.txt')
    # train,valid,test三个数据集文件夹的建立
    train_fold = os.path.join(out_path,'train')
    valid_fold = os.path.join(out_path,'valid')
    test_fold = os.path.join(out_path,'test')
    for fold in [valid_fold,test_fold,train_fold]:
        if not os.path.exits(fold):
            os.mkdir(fold)
    
    # 移动train,valid,test三个数据集所需要的文件
    move_files(gcommands_fold,test_fold,test_path)
    move_files(gcommands_fold,valid_fold,validation_path)
    create_train_fold(gcommands_fold,train_fold,test_fold)

if __name__ == '__main__':
    parser = argparser.ArgumentParser(description = 'Make speech commands dataset.')
    parser.add_argument('-in_path',default='train',help='the path to the root folder of the speech commands dataset.')
    parser.add_argument('-out_path',default='data',help='the path where to save the files splitted to folders.')
    args = parser.parse_args()
    make_dataset(args.gcommands_fold,args.out_path)
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    