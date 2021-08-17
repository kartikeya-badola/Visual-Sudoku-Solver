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
parser.add_argument('--test_query', default = '/content/Assignment 2/visual_sudoku/train/query/', type=str, help='path to test_query')


args = parser.parse_args()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

dataset = datasets.ImageFolder('binary_Data',transform)
# valset = datasets.ImageFolder('/content/clusters',transform)

trainset,valset = random_split(dataset,[int(0.8*len(dataset)),len(dataset)-int(0.8*len(dataset))])



trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=128,
                                          shuffle=True, num_workers=2)

num_classes = 2
image_size = 28
channels=3

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

net = LeNet(2).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

dataloaders = {'train':trainloader,'val':valloader}

import time
import copy

def train_model(model,optimizer,loss_fn, num_epochs=100,patience=100):
    clock=0
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


net = train_model(net,optimizer,loss_fn,1,1)
torch.save(net,'binary.pth')

# net=net.to(device).eval()
net.eval()
sudoku=[]

img_list = []
for l in tqdm(sorted(os.listdir(args.train_path+'/query/'))):
  pil_img = Image.open(args.train_path+'/query/'+l)
  tensor = transforms.ToTensor()(pil_img)
  img_list.append(tensor.unsqueeze(0))

imgs = torch.cat(img_list,dim=0)
patches = imgs.unfold(2, 28, 28).unfold(3, 28, 28)
patches = patches.reshape(10000,1,64,28,28).permute(0,2,1,3,4)
for te in tqdm(patches):
#   te.unsqueeze(0)
  te=te.repeat(1,3,1,1)
  _,pred = net(te.to(device)).max(1)
  sudoku.append(pred.cpu().unsqueeze(0))


query_sudoku_mask=torch.cat(sudoku,dim=0)

torch.save(query_sudoku_mask,'query_sudoku_mask.pth')
sudoku=[]

img_list = []
for l in tqdm(sorted(os.listdir(args.test_query))):
  pil_img = Image.open(args.test_query+l)
  tensor = transforms.ToTensor()(pil_img)
  img_list.append(tensor.unsqueeze(0))

imgs = torch.cat(img_list,dim=0)
patches = imgs.unfold(2, 28, 28).unfold(3, 28, 28)
patches = patches.reshape(10000,1,64,28,28).permute(0,2,1,3,4)
for te in tqdm(patches):
#   te.unsqueeze(0)
  te=te.repeat(1,3,1,1)
  _,pred = net(te.to(device)).max(1)
  sudoku.append(pred.cpu().unsqueeze(0))


test_query_sudoku_mask=torch.cat(sudoku,dim=0)

torch.save(test_query_sudoku_mask,'test_query_sudoku_mask.pth')

