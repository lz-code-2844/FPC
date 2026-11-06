import torch
from torch import nn
import pandas as pd
import os
from torch.utils.data import Dataset,DataLoader
import numpy as np
import pickle

def read_ti(fn):
    data = []
    with open(fn) as f:
        file = f.readlines()
        for line in file:
            line = line.strip()
            dd = line.split()
            temp = list(dd)
            data.append(temp)
    data = np.delete(data, 0, axis=0)
    data = np.array(data, dtype='float64')
    return data

def read_csv(fn):
    data = []
    for i,line in enumerate(open(fn)):
        try:
            nd = line.split(',')
            _ = float(nd[0])+float(nd[1])
            data.append(nd)
        except:
            continue
    data = np.array(data,dtype='float64')
    return data

def norm_data(datapath):
    min_val = np.array([1e20,])
    max_val = np.array([-1e20,])
    num_data = 0
    num_points = []
    f_list = os.listdir(datapath)
    f_list.sort(key=lambda x: int(x[:-4]))
    for f in f_list:
        if f[-3:] == 'txt':
            data = read_ti(os.path.join(datapath,f))
            if min_val.shape[0] == 1:
                min_val = min_val.repeat(data.shape[1])
                max_val = max_val.repeat(data.shape[1])
            min_val = np.min([min_val,data.min(axis=0)],axis=0)
            max_val = np.max([max_val,data.max(axis=0)],axis=0)
            num_data += 1
            num_points.append(data.shape[0])
            print(min_val)
    return min_val,max_val, num_data,num_points

class pointdata(Dataset):
    def __init__(self,input_path,output_path,data_info,point_number):
        self._input_path  = input_path
        self._label = read_csv(output_path)
        self._data_info = data_info
        self.input_norm_d = data_info['input_max'] - data_info['input_min'] 
        self.point_number = point_number
        self._data_info['num_data'] = self._label.shape[0]


    def __len__(self):
        return self._data_info['num_data']
    
    def __getitem__(self,index):
        #fileNo = int(self._label[index,0])
        fileNo=0
        data=read_ti(os.path.join(self._input_path,'%d.txt'%fileNo))
        np.random.shuffle(data)
        label=data[:,[4,5,6,7]]
        for i in range(label.shape[1]):
            col = label[:, i]
            label[:, i] = (col - col.min()) / (col.max() - col.min())
        for i in range(1, 4):
            col = data[:, i]
            data[:, i] = (col - col.min()) / (col.max() - col.min())
        data=data[:,0:4]
        a=self._label[index,0]
        return data.T,label.T


