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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from PIL import Image
from tqdm import tqdm
parser = argparse.ArgumentParser(description='COL 870 Assignment 2.3')
parser.add_argument('--train_path', default='/content/Assignment 2/visual_sudoku/train', type=str, help='path to train images')
parser.add_argument('--test_query', default = '/content/Assignment 2/visual_sudoku/train/query/', type=str, help='path to test_query')
parser.add_argument('--sample_images', default = '/content/Assignment 2/sample_images.npy', type=str, help='path to sample images.npy')
args = parser.parse_args()

img_list = []
for l in tqdm(sorted(os.listdir(args.train_path+'/target'))):
  pil_img = Image.open(args.train_path+'/target/'+l)
  tensor = transforms.ToTensor()(pil_img)
  img_list.append(tensor.unsqueeze(0))

imgs = torch.cat(img_list,dim=0)
patches = imgs.unfold(2, 28, 28).unfold(3, 28, 28)
patches = patches.reshape(10000,1,64,28,28).permute(0,2,1,3,4)

class TripletDataset(data.Dataset):
    def __init__(self):
        self.target = patches.reshape(640000,28,28)
        self.augment = transforms.Compose([transforms.ToPILImage(),
                             transforms.RandomAffine(degrees=15, translate=(0.1,0.1), scale=(1.2,1.2)),
                             transforms.ToTensor()
                            ])
    
    def __len__(self):
        return 14 * self.target.shape[0]
    
    def __getitem__(self, S):
        
        i, j = S // 14, S % 14
        image = self.target[i]

        if j < 7:
            idx = 8 * (i // 8) + (i + j + 1) % 8
        else:
            idx = 64 * (i // 64) + (i + 8 * (j - 6)) % 64

        negative = self.target[idx]
        return image.unsqueeze(0), self.augment(image), negative.unsqueeze(0)

trainset = TripletDataset()
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                          shuffle=True, num_workers=4)
class residual_block(nn.Module):
  def __init__(self,in_channels: int, downsample: bool,normalization):
    super(residual_block,self).__init__()
    self.downsample=downsample
    self.perform_norm=not (normalization is None)
    print('using norm?',self.perform_norm)
    if downsample:
      self.identity_downsample = nn.Conv2d(in_channels=in_channels,out_channels=in_channels*2,kernel_size=1,stride=2,padding=0)
      self.layer1 = nn.Conv2d(in_channels=in_channels,out_channels=in_channels*2,kernel_size=3,stride=2,padding=1)
      self.layer2 = nn.Conv2d(in_channels=in_channels*2,out_channels=in_channels*2,kernel_size=3,stride=1,padding=1)
      if self.perform_norm:
        self.norm1 = normalization(in_channels*2)
        self.norm2 = normalization(in_channels*2)
    else:
      self.layer1 = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,padding=1)
      self.layer2 = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,padding=1)
      if self.perform_norm:
        self.norm1 = normalization(in_channels)
        self.norm2 = normalization(in_channels)
  def forward(self,x):
    residual=x
    if self.downsample:
      residual=self.identity_downsample(x)
    if self.perform_norm:
      x = self.norm2(self.layer2(nn.ReLU()(self.norm1(self.layer1(x)))))
      
    else:
      x = self.layer2(nn.ReLU()(self.layer1(x)))
    x = nn.ReLU()(residual+x)
    return x
class block_of_residual_blocks(nn.Module):
  def __init__(self,num_blocks: int, in_channels: int ,firstblock: bool,normalization):
    super(block_of_residual_blocks,self).__init__()
    self.blocks = []
    if firstblock:
      self.blocks.append(residual_block(in_channels=in_channels,downsample=False,normalization=normalization))
      for i in range(num_blocks-1):
        self.blocks.append(residual_block(in_channels=in_channels,downsample=False,normalization=normalization))
    else:
      self.blocks.append(residual_block(in_channels=in_channels,downsample=True,normalization=normalization))
      for i in range(num_blocks-1):
        self.blocks.append(residual_block(in_channels=2*in_channels,downsample=False,normalization=normalization))
    self.blocks = nn.ModuleList(self.blocks)
    
  def forward(self,x):
    for block in self.blocks:
      x=block(x)

    return x
class my_resnet(nn.Module):
  def __init__(self,num_blocks: int,normalization):
    super(my_resnet,self).__init__()
    self.input_layer = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1)
    
    self.perform_norm = not (normalization is None)
    print('using input norm?',self.perform_norm)
    if self.perform_norm:
      self.input_norm = normalization(16)
    self.block1 = block_of_residual_blocks(num_blocks=num_blocks,in_channels=16,firstblock=True,normalization=normalization)

    self.block2 = block_of_residual_blocks(num_blocks=num_blocks,in_channels=16,firstblock=False,normalization=normalization)

    self.block3 = block_of_residual_blocks(num_blocks=num_blocks,in_channels=32,firstblock=False,normalization=normalization)

    # self.fc = nn.Linear(8*8,num_classes)
    self.pool = nn.AdaptiveAvgPool2d((1, 1))
  def forward(self,x):
    x= self.input_layer(x)
    if self.perform_norm:
      x = self.block3(self.block2(self.block1(nn.ReLU()(self.input_norm(x)))))
    else:
      x = self.block3(self.block2(self.block1(nn.ReLU()(x))))
    x = self.pool(x)
    x = torch.flatten(x, 1)
    return F.sigmoid(x)
model = my_resnet(4,nn.BatchNorm2d).to(device)
loss = nn.TripletMarginLoss(margin=2.0)
optimizer = optim.Adam(model.parameters())


for epoch in range(1):

  print('Epoch {}/{}'.format(epoch, 9))
  print('-' * 10)


  # Each epoch has a training and validation phase
  model.train()

  running_loss = 0.0
  running_corrects = 0 
  running_total = 0
  num_batches=0
  # predlist=torch.zeros(0,dtype=torch.long, device='cpu')
  # labellist=torch.zeros(0,dtype=torch.long, device='cpu')


  # Iterate over data.
  # for inputs, labels in dataloaders[phase]:
  for i,(a,p,n) in tqdm(enumerate(trainloader)):
      if i==5000:
        break

      # im = im.to(device)
      # labels = labels.to(device)
      emb_a = model(a.to(device))
      emb_p = model(p.to(device))
      emb_n = model(n.to(device))
      
      loss_value = loss(emb_a,emb_p,emb_n)
      optimizer.zero_grad()
      loss_value.backward()
      optimizer.step()
          

      # statistics

      running_loss += loss_value.item()
      running_total += len(a)

      if i%1000==1:
        print('loss:',running_loss/i)
        torch.save(model,'triplet_loss.pth')

  # if phase == 'train':
      # scheduler.step()

  # epoch_loss = running_loss/i
  # epoch_acc = running_corrects/running_total

  # print('{} Loss: {:.4f} acc: {:.4f}'.format(epoch_loss, epoch_acc))
# model = torch.load('triplet_loss.pth').to(device)
torch.save(model,'triplet_loss.pth')
imgs = np.load(args.sample_images)
classes = torch.Tensor(imgs[1:-1]/255.0)
model.eval()
gt_emb = model(classes.unsqueeze(1).to(device))
gt_emb = gt_emb.unsqueeze(0).repeat(64,1,1)

sudoku=[]
for p in tqdm(patches):
  p_emb = model(p.to(device))
  p_emb = p_emb.unsqueeze(1).repeat(1,8,1)
  # print(p_emb.shape,gt_emb.shape)
  x = torch.norm(gt_emb-p_emb,dim=2)
  # print(x)
  _,pred = x.min(dim=1)
  
  # assert 1==2
  sudoku.append(pred.cpu().reshape(8,8))
  

for i,s in enumerate(sudoku):
  sudoku[i]=s.unsqueeze(0)
for i,s in enumerate(sudoku):
  sudoku[i]=s+1
sudoku_dataset = torch.cat(sudoku,dim=0)
torch.save(sudoku_dataset,'target_sudoku.pth')

for i in range(1,9):
    os.mkdir('gan_Dataset/'+str(i))

for i in tqdm(range(len(sudoku))):
    s= sudoku[i].reshape(64)
    imgs = patches[i]
    for j in range(64):
        torchvision.utils.save_image(1-imgs[j],'gan_Dataset/'+str(s[j].item())+'/'+str(i)+'_'+str(j)+'.png')




img_list = []
for l in tqdm(sorted(os.listdir(args.test_query))):
  pil_img = Image.open(args.test_query+l)
  tensor = transforms.ToTensor()(pil_img)
  img_list.append(tensor.unsqueeze(0))

imgs = torch.cat(img_list,dim=0)
patches = imgs.unfold(2, 28, 28).unfold(3, 28, 28)
patches = patches.reshape(10000,1,64,28,28).permute(0,2,1,3,4)

sudoku=[]
for p in tqdm(patches):
  p_emb = model(p.to(device))
  p_emb = p_emb.unsqueeze(1).repeat(1,8,1)
  # print(p_emb.shape,gt_emb.shape)
  x = torch.norm(gt_emb-p_emb,dim=2)
  # print(x)
  _,pred = x.min(dim=1)
  
  # assert 1==2
  sudoku.append(pred.cpu().reshape(8,8))

for i,s in enumerate(sudoku):
  sudoku[i]=s.unsqueeze(0)
for i,s in enumerate(sudoku):
  sudoku[i]=s+1
sudoku_dataset = torch.cat(sudoku,dim=0)
torch.save(sudoku_dataset,'test_query_sudoku.pth')