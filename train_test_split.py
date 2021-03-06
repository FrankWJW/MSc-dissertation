#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
utility function:

Spliting train and test + normalisation + making window

input:  data, target, len_of_trainset, time_interval

output: trainData, testData, validateData
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
# from keras.layers import Dense, Dropout, Activation, LSTM, Convolution1D, MaxPooling1D, Flatten
# from keras.models import Sequential
import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')
from torchbearer import Trial
import os

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchbearer
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.datasets import MNIST
from torchbearer import Trial
import torchvision.transforms as transforms


# In[2]:


class utility_fun():
    
    
    def __init__(self, data, len_of_trainset = 2412, time_interval = 100, batch_size = 32):
        # print('start')
        self.X = data[0]
        self.y = data[1]
        self.len_of_trainset = len_of_trainset
        self.time_interval = time_interval
        
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.batch_size = batch_size
    def norm(self, flag):

        len_of_trainset = self.len_of_trainset

        y1 = self.y[:len_of_trainset]
        y2 = self.y[len_of_trainset:]

        X1 = self.X[:len_of_trainset,:]
        X2 = self.X[len_of_trainset:,:]
        
        X1= (X1- np.min(X1,axis=0))/(np.max(X1, axis=0)-np.min(X1,axis=0))
        X2= (X2- np.min(X2,axis=0))/(np.max(X2, axis=0)-np.min(X2,axis=0))
        
        if flag is 'classifier':
            print(flag)
        if flag is 'regressor':
            print(flag)
            y1= (y1- min(y1))/(max(y1)-min(y1))
            y2= (y2- min(y2))/(max(y2)-min(y2))
            self.y = np.concatenate((y1,y2))
            
        self.X = np.concatenate((X1,X2) , axis = 0)
        
#         print(self.X.shape, self.y.shape)
        
    def sepera_time_step(self):
        time_steps= self.time_interval
        X = self.X
        y = self.y
        X_new= np.zeros((X.shape[0] - time_steps +1, time_steps, X.shape[1]))
        y_new= np.zeros((y.shape[0] -time_steps +1,))
        for ix in range(X_new.shape[0]):
            for jx in range(time_steps):
                X_new[ix, jx, :]= X[ix +jx, :]
            y_new[ix]= y[ix + time_steps -1]
#         print (X_new.shape, y_new.shape)
        self.X = X_new
        self.y = y_new
        
    def test_train_split(self):
        split = self.len_of_trainset
#         X_train = X_new[:split]
#         X_test = X_new[split:]
        
        self.X_train = self.X[:split]
        self.X_test = self.X[split:]

        self.y_train = self.y[:split]
        self.y_test = self.y[split:]

#         print (X_train.shape, y_train.shape)
#         print (X_test.shape, y_test.shape)


    def build_dataloader(self, flag = 'mlp'):
        #convert to torch
        trainData = torch.from_numpy(self.X_train)
        testData = torch.from_numpy(self.y_train)
        validateData_sample = torch.from_numpy(self.X_test)
        validateData_target = torch.from_numpy(self.y_test)
        
        lens = self.len_of_trainset
        time_steps = self.time_interval
        
        
        trainData = trainData.view(lens,-1,time_steps,11)
        testData = testData.view(lens,-1)
        validateData_sample = validateData_sample.view(validateData_sample.shape[0],-1,time_steps,11)
#         print(trainData.shape,testData.shape,validateData.shape)
        
        # if using mlp
        if flag == 'mlp' or 'lstm':
            train = TensorDataset(trainData.view(lens, -1).float(), testData.view(lens, -1).float())
            len_test = validateData_sample.shape[0]
            test = TensorDataset(validateData_sample.view(len_test, -1).float(), validateData_target.view(len_test, -1).float())
        
        # if using conv 1d
        if flag == '1dconv':
            train = TensorDataset(trainData, testData)
            len_test = validateData_sample.shape[0]
            test = TensorDataset(validateData_sample.float(), validateData_target.view(len_test, -1).float())
        
        trainloader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        testloader = DataLoader(test, batch_size=self.batch_size, shuffle=True)
        
        
        return trainloader, testloader, validateData_sample, validateData_target, trainData, testData
        


# testing

