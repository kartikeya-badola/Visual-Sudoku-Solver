import os
import random
import logging
import argparse
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm

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
from torchvision import datasets, models, transforms
import argparse

parser = argparse.ArgumentParser(description='COL 870 Assignment 2.1')
parser.add_argument('--train_path', default='/content/Assignment 2/visual_sudoku/train', type=str, help='path to train images')
parser.add_argument('--test_query', default = '/content/Assignment 2/visual_sudoku/train/query', type=str, help='path to test_query')
# parser.add_argument('--target_path', default='target9k.npy', type=str, help='path to target9k.npy')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed = 870)

from torch.utils.data import random_split
from torchvision import datasets, models, transforms
transform = transforms.Compose([
    transforms.ToTensor()])

dataset = datasets.ImageFolder('new_clusters2',transform)
# valset = datasets.ImageFolder('/content/clusters',transform)

trainset,valset = random_split(dataset,[int(0.8*len(dataset)),len(dataset)-int(0.8*len(dataset))])



trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=4)
valloader = torch.utils.data.DataLoader(valset, batch_size=128,
                                          shuffle=True, num_workers=4)

num_classes = 9 
image_size = 28
channels=3

# class residual_block(nn.Module):
#   def __init__(self,in_channels: int, downsample: bool,normalization):
#     super(residual_block,self).__init__()
#     self.downsample=downsample
#     self.perform_norm=not (normalization is None)
#     print('using norm?',self.perform_norm)
#     if downsample:
#       self.identity_downsample = nn.Conv2d(in_channels=in_channels,out_channels=in_channels*2,kernel_size=1,stride=2,padding=0)
#       self.layer1 = nn.Conv2d(in_channels=in_channels,out_channels=in_channels*2,kernel_size=3,stride=2,padding=1)
#       self.layer2 = nn.Conv2d(in_channels=in_channels*2,out_channels=in_channels*2,kernel_size=3,stride=1,padding=1)
#       if self.perform_norm:
#         self.norm1 = normalization(in_channels*2)
#         self.norm2 = normalization(in_channels*2)
#     else:
#       self.layer1 = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,padding=1)
#       self.layer2 = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,padding=1)
#       if self.perform_norm:
#         self.norm1 = normalization(in_channels)
#         self.norm2 = normalization(in_channels)
#   def forward(self,x):
#     residual=x
#     if self.downsample:
#       residual=self.identity_downsample(x)
#     if self.perform_norm:
#       x = self.norm2(self.layer2(nn.ReLU()(self.norm1(self.layer1(x)))))
      
#     else:
#       x = self.layer2(nn.ReLU()(self.layer1(x)))
#     x = nn.ReLU()(residual+x)
#     return x
# class block_of_residual_blocks(nn.Module):
#   def __init__(self,num_blocks: int, in_channels: int ,firstblock: bool,normalization):
#     super(block_of_residual_blocks,self).__init__()
#     self.blocks = []
#     if firstblock:
#       self.blocks.append(residual_block(in_channels=in_channels,downsample=False,normalization=normalization))
#       for i in range(num_blocks-1):
#         self.blocks.append(residual_block(in_channels=in_channels,downsample=False,normalization=normalization))
#     else:
#       self.blocks.append(residual_block(in_channels=in_channels,downsample=True,normalization=normalization))
#       for i in range(num_blocks-1):
#         self.blocks.append(residual_block(in_channels=2*in_channels,downsample=False,normalization=normalization))
#     self.blocks = nn.ModuleList(self.blocks)
    
#   def forward(self,x):
#     for block in self.blocks:
#       x=block(x)

#     return x
# class my_resnet(nn.Module):
#   def __init__(self,num_blocks: int,normalization,num_classes:int):
#     super(my_resnet,self).__init__()
#     self.input_layer = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1)
    
#     self.perform_norm = not (normalization is None)
#     print('using input norm?',self.perform_norm)
#     if self.perform_norm:
#       self.input_norm = normalization(16)
#     self.block1 = block_of_residual_blocks(num_blocks=num_blocks,in_channels=16,firstblock=True,normalization=normalization)

#     self.block2 = block_of_residual_blocks(num_blocks=num_blocks,in_channels=16,firstblock=False,normalization=normalization)

#     self.block3 = block_of_residual_blocks(num_blocks=num_blocks,in_channels=32,firstblock=False,normalization=normalization)

#     self.fc = nn.Linear(8*8,num_classes)
#     self.pool = nn.AdaptiveAvgPool2d((1, 1))
#   def forward(self,x):
#     x= self.input_layer(x)
#     if self.perform_norm:
#       x = self.block3(self.block2(self.block1(nn.ReLU()(self.input_norm(x)))))
#     else:
#       x = self.block3(self.block2(self.block1(nn.ReLU()(x))))
#     x = self.pool(x)
#     x = torch.flatten(x, 1)
#     return self.fc(x)

# net = my_resnet(4,nn.BatchNorm2d,9).to(device)


class LeNet(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,6,5), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6,16,5), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256,120), nn.ReLU(),
            nn.Linear(120,84), nn.ReLU(),
            nn.Linear(84,num_classes)
        )
        # self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, X):
        return self.fc(torch.flatten(self.conv(X),-3,-1))
    
    # def criterion(self, Y, y):
    #     return self.cross_entropy_loss(Y, y)
    
    # def predict(self, X):
    #     return torch.argmax(self(X), dim=-1)

net = LeNet(9).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

dataloaders = {'train':trainloader,'val':valloader}

import time
import copy

def train_model(model,optimizer,loss_fn, num_epochs=100,patience=100):
    clock=1
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100
    # best_acc=0

    for epoch in range(num_epochs):
        if(clock<patience):
          clock=clock+1
          print('Epoch {}/{}'.format(epoch, num_epochs - 1))
          print('-' * 10)
          for param in optimizer.param_groups:
            print('learning rate is: ', param['lr'])
          
          activations_list = []

          

          # Each epoch has a training and validation phase
          for phase in ['train','val']:
              if phase == 'train':
                  model.train()  # Set model to training mode
              else:
                  model.eval()   # Set model to evaluate mode

              running_loss = 0.0
              running_corrects = 0 
              running_total = 0
              num_batches=0
              # predlist=torch.zeros(0,dtype=torch.long, device='cpu')
              # labellist=torch.zeros(0,dtype=torch.long, device='cpu')

  
              # Iterate over data.
              # for inputs, labels in dataloaders[phase]:
              for i,(im,labels) in tqdm(enumerate(dataloaders[phase])):
        
                  im = im.to(device)
                  labels = labels.to(device)
                  
                  # forward
                  # track history if only in train
                  with torch.set_grad_enabled(phase == 'train'):
                      logits = model(im)
                        

                      _, preds = torch.max(logits, 1)
                      loss = loss_fn(logits, labels)


                      # print(inputs.shape)
                      


                      # backward + optimize only if in training phase
                      if phase == 'train':
                          optimizer.zero_grad()
                          loss.backward()
                          optimizer.step()
                      

                  # statistics
    
                  running_loss += loss.item()
                  # predlist=torch.cat([predlist,preds.view(-1).cpu()])
                  # labellist=torch.cat([labellist,labels.data.view(-1).cpu()])
                  running_corrects += torch.sum(preds == labels.data)
                  running_total += len(labels)
                  num_batches+=1

                  if num_batches%1000==0:
                    print('accuracy:',running_corrects/running_total)
              # if phase == 'train':
                  # scheduler.step()


            



              epoch_loss = running_loss/num_batches
              epoch_acc = running_corrects/running_total

              print('{} Loss: {:.4f} acc: {:.4f}'.format(
                  phase, epoch_loss, epoch_acc))


              # deep copy the model
              if phase == 'val' and epoch_loss < best_loss:
                  print('BEST MODEL FOUND!!!')
                  clock=1
                  best_loss = epoch_loss
                  torch.save(model,'best_model.pth')
                  best_model_wts = copy.deepcopy(model.state_dict())
              # if phase == 'val' and epoch_acc > best_acc:
              #     print('BEST MODEL FOUND!!!')
              #     clock=1
              #     best_acc = epoch_acc
              #     best_model_wts = copy.deepcopy(model.state_dict())


        else:
          print('EARLY STOPPING!!!!!')
          break

        # print()
    
    

    # print(tensor_activations.shape)

    # print(Q1,Q2,Q3,Q4)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    # print('Best val acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


net = train_model(net,optimizer,loss_fn,5,5)


img_list = []
query_path = os.path.join(args.train_path,'query/')
for l in tqdm(sorted(os.listdir(query_path))):
  pil_img = Image.open(query_path+l)
  tensor = transforms.ToTensor()(pil_img)
  img_list.append(tensor.unsqueeze(0))

imgs = torch.cat(img_list,dim=0)
patches = imgs.unfold(2, 28, 28).unfold(3, 28, 28)
patches = patches.reshape(len(img_list),1,64,28,28).permute(0,2,1,3,4)
patches=1-patches

sudoku=[]
net=net.to(device).eval()
for p in tqdm(patches):
  p=p.repeat(1,3,1,1)
  _,pred = net(p.to(device)).max(1)
  sudoku.append(pred.cpu().unsqueeze(0))

query_sudoku=torch.cat(sudoku,dim=0)

img_list = []
query_path = os.path.join(args.train_path,'target/')
for l in tqdm(sorted(os.listdir(query_path))):
  pil_img = Image.open(query_path+l)
  tensor = transforms.ToTensor()(pil_img)
  img_list.append(tensor.unsqueeze(0))

imgs = torch.cat(img_list,dim=0)
patches = imgs.unfold(2, 28, 28).unfold(3, 28, 28)
patches = patches.reshape(len(img_list),1,64,28,28).permute(0,2,1,3,4)
patches=1-patches


  
sudoku=[]
net=net.to(device).eval()
for p in tqdm(patches):
  p=p.repeat(1,3,1,1)
  _,pred = net(p.to(device)).max(1)
  sudoku.append(pred.cpu().unsqueeze(0)+1)

target_sudoku=torch.cat(sudoku,dim=0)

torch.save(query_sudoku,'query_sudoku.pth')
torch.save(target_sudoku,'target_sudoku.pth')

img_list = []
for l in tqdm(sorted(os.listdir(args.test_query))):
  pil_img = Image.open(query_path+l)
  tensor = transforms.ToTensor()(pil_img)
  img_list.append(tensor.unsqueeze(0))

imgs = torch.cat(img_list,dim=0)
patches = imgs.unfold(2, 28, 28).unfold(3, 28, 28)
patches = patches.reshape(len(img_list),1,64,28,28).permute(0,2,1,3,4)
patches=1-patches

sudoku=[]
net=net.to(device).eval()
for p in tqdm(patches):
  p=p.repeat(1,3,1,1)
  _,pred = net(p.to(device)).max(1)
  sudoku.append(pred.cpu().unsqueeze(0))

test_query_sudoku=torch.cat(sudoku,dim=0)

torch.save(test_query_sudoku,'test_query.pth')