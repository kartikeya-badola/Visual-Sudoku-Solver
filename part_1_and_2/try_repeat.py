from collections import Counter
import itertools
from functools import reduce
from sklearn.metrics import f1_score, accuracy_score
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import time
import copy
from tqdm import tqdm
import argparse
import torch.nn.functional as F
from torch.nn import LSTMCell
from torch.utils.data import random_split
from PIL import Image
import os
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms, utils
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class G(nn.Module):
  def __init__(self,n_classes,hidden_dim):
    super(G,self).__init__()
    self.y = nn.Sequential(nn.Linear(n_classes,1000),nn.LeakyReLU(0.2))
    self.z = nn.Sequential(nn.Linear(hidden_dim,200),nn.BatchNorm1d(200),nn.LeakyReLU(0.2))
    self.zy = nn.Sequential(nn.Linear(1200,1200),nn.BatchNorm1d(1200),nn.LeakyReLU(0.2),nn.Linear(1200,784),nn.Sigmoid())
    self.n_classes = n_classes
  def forward(self,noise,labels):
    bsz = labels.shape[0]
    y = self.y(torch.zeros(bsz,self.n_classes).to(device).scatter_(1,labels.unsqueeze(1),1))
    z = self.z(noise)
    zy = torch.cat([z,y],dim=1)
    return self.zy(zy)

class maxout_layer_dropout(nn.Module):
  def __init__(self,input_dim,output_dim,num_pieces):
    super().__init__()
    self.id = input_dim
    self.od = output_dim
    self.np = num_pieces
    self.layer = nn.Linear(input_dim,output_dim*num_pieces)
  def forward(self,x):
    # bsz,_= x.shape
    z = self.layer(x).view(-1,self.np,self.od)
    # print(z.shape)
    z,_ = torch.max(z,dim=1)
    # z = z.squeeze(0)
    return F.dropout(z,p=0.5,training=self.training)

class D(nn.Module):
  def __init__(self,n_classes):
    super(D,self).__init__()
    self.y = maxout_layer_dropout(n_classes,50,5)
    self.z = maxout_layer_dropout(784,240,5)
    self.zy = maxout_layer_dropout(290,240,4)
    self.final = nn.Sequential(nn.Linear(240,1),nn.Sigmoid())
    # self.final = nn.Linear(240,1)
    self.n_classes = n_classes
  def forward(self,img,labels):
    bsz = labels.shape[0]
    y = self.y(torch.zeros(bsz,self.n_classes).to(device).scatter_(1,labels.unsqueeze(1),1))
    z = self.z(img)
    zy = self.zy(torch.cat([z,y],dim=1))
    return self.final(zy)



g = torch.load('G1.pth').to(device)
d = torch.load('D1.pth').to(device)

g.eval()


# z =  torch.FloatTensor(9000,100).uniform_(0,1).to(device)
# # Get labels ranging from 0 to n_classes for n rows
# labels = np.array([num for num in range(9) for _ in range(1000)])
# labels = torch.Tensor(labels).long().to(device)
# gen_imgs = g(z, labels).cpu().detach()
# gen_imgs = gen_imgs.reshape(9000,1,28,28)


# z = torch.FloatTensor(9000*repeats,100).uniform_(0,1).to(device)
# labels = np.array([num for num in range(9) for _ in range(1000*repeats)])
# labels = torch.Tensor(labels).long().to(device)
# gen_imgs = g(z,labels).cpu().detach()
# gen_imgs = gen_imgs.reshape(9000*repeats, 1, 28, 28)



# for i in range(9):
#   for j in range(1000*repeats):
#     img = gen_imgs[i*1000*repeats + j].unsqueeze(0)
#     torchvision.utils.save_image(img, 'new_clusters2/' + str(i) + '/' + str(i) + '_' + str(j) + '.png')

os.mkdir('new_clusters2')

for i in range(9):
   os.mkdir('new_clusters2/' + str(i))

repeats = 5 

for r in range(repeats):
    z = torch.FloatTensor(9000,100).uniform_(0,1).to(device)
    labels = np.array([num for num in range(9) for _ in range(1000)])
    labels = torch.Tensor(labels).long().to(device)
    gen_imgs = g(z,labels).cpu().detach()
    gen_imgs = gen_imgs.reshape(9000, 1, 28, 28)

    for i in range(9):
        for j in range(1000):
            img = gen_imgs[i*1000 + j].unsqueeze(0)
            torchvision.utils.save_image(img, 'new_clusters2/' + str(i) + '/' + str(i) + '_' + str(1000*r + j) + '.png')