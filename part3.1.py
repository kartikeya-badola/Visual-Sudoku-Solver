import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import torch.nn.functional as F
import copy
from sklearn.utils import shuffle
import torch.utils.data as data
from torch.utils.data import random_split
from tqdm import tqdm
from PIL import Image
import os
import argparse
import random

parser = argparse.ArgumentParser(description='COL 870 Assignment 2.3')
parser.add_argument('--train_path', default='/content/Assignment 2/visual_sudoku/train', type=str, help='path to train images')


args = parser.parse_args()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
img_list = []
for l in tqdm(sorted(os.listdir(args.train_path+'/target'))):
  pil_img = Image.open(args.train_path+'/target/'+l)
  tensor = transforms.ToTensor()(pil_img)
  img_list.append(tensor.unsqueeze(0))

imgs = torch.cat(img_list,dim=0)
patches = imgs.unfold(2, 28, 28).unfold(3, 28, 28)
patches = patches.reshape(10000,1,64,28,28).permute(0,2,1,3,4)

qimg_list = []
for l in tqdm(sorted(os.listdir(args.train_path+'/query'))):
  pil_img = Image.open(args.train_path+'/query/'+l)
  tensor = transforms.ToTensor()(pil_img)
  qimg_list.append(tensor.unsqueeze(0))

qimgs = torch.cat(qimg_list,dim=0)
qpatches = qimgs.unfold(2, 28, 28).unfold(3, 28, 28)
qpatches = qpatches.reshape(10000,1,64,28,28).permute(0,2,1,3,4)

qpatches = qpatches.reshape(640000,28*28)
tpatches = patches.reshape(640000,28*28)
img_list=[]
qimg_list=[]

os.mkdir('binary_Data')
os.mkdir('binary_Data/0')
os.mkdir('gan_Dataset/')
os.mkdir('gan_Dataset/0')

for i in tqdm(range(6400)):
  qp = qpatches[100*i:100*(i+1)]
  tp = tpatches[100*i:100*(i+1)]
  zero_mask = 1-((qp==tp).sum(1)==28*28).long()
  zeros = qp[zero_mask.bool(),:]
  zeros = zeros.reshape(len(zeros),1,28,28)
  for j in range(len(zeros)):
    torchvision.utils.save_image(zeros[j].unsqueeze(0),'binary_Data/0/'+str(i)+'_'+str(j)+'.png')
    torchvision.utils.save_image(1-zeros[j].unsqueeze(0),'gan_Dataset/0/'+str(i)+'_'+str(j)+'.png')


os.mkdir('binary_Data/1')
tp = tpatches[torch.randperm(len(os.listdir('binary_Data/0')))]
tp = tp.reshape(len(tp),1,28,28)
for i in tqdm(range(len(tp))):
  torchvision.utils.save_image(tp[i].unsqueeze(0),'binary_Data/1/'+str(i)+'.png')

