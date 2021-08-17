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
import os
from torch.nn import LSTMCell
from torch.utils.data import random_split
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='COL 870 Assignment 2.3')
parser.add_argument('--test_query', default = '/content/Assignment 2/visual_sudoku/train/query/', type=str, help='path to test_query')
parser.add_argument('--output_csv', default = 'output.csv', type=str, help='path to output.csv')
args = parser.parse_args()
# query_sudoku = torch.load("/content/query_sudoku.pth")
target_sudoku = torch.load('target_sudoku.pth')
query_sudoku_mask = torch.load('query_sudoku_mask.pth')
query_sudoku_mask = query_sudoku_mask.reshape(-1,8,8)
query_sudoku = target_sudoku*query_sudoku_mask
test_query_sudoku = torch.load('test_query_sudoku.pth')
test_query_sudoku_mask  = torch.load('test_query_sudoku_mask.pth')
test_query_sudoku_mask = test_query_sudoku_mask.reshape(-1,8,8)
test_query_sudoku = test_query_sudoku*test_query_sudoku_mask

class sudokuDataset(Dataset):
  def __init__(self, tensors):

    self.data_size = len(query_sudoku)
    self.query = query_sudoku.long().reshape(self.data_size,64)
    self.answer = target_sudoku.long().reshape(self.data_size,64)
    

  def __getitem__(self, i):
      return self.query[i], self.answer[i]

  def __len__(self):
      return self.data_size

dataset = sudokuDataset((query_sudoku, target_sudoku))
train_data,val_data,test_data = random_split(dataset,[int(0.7*len(dataset)),int(0.2*len(dataset)),len(dataset)-int(0.7*len(dataset))-int(0.2*len(dataset))])
dataloaders={}
dataloaders['train'] = torch.utils.data.DataLoader(train_data,batch_size=64, shuffle=True, num_workers=2)
dataloaders['val'] = torch.utils.data.DataLoader(val_data,batch_size=64, shuffle=False, num_workers=2)
dataloaders['test'] = torch.utils.data.DataLoader(test_data,batch_size=64, shuffle=False,num_workers=2)

sudoku_mask = torch.zeros(64,8,8)
rows = torch.zeros(64)
columns = torch.zeros(64)
for i in range(64):
  row=i//8
  rows[i]=row
  column=i%8
  columns[i]=column
  # print(row,column)
  sudoku_mask[i,row,:]=1 
  sudoku_mask[i,:,column]=1
  block_row = row//2
  block_column = column//4
  sudoku_mask[i,2*block_row:2*(block_row+1),4*block_column:4*(block_column+1)]=1
  sudoku_mask[i][row][column]=0

sudoku_mask = sudoku_mask.bool()
class RRN(nn.Module):
  def __init__(self,emb_dim=16,num_steps=25,mlp_dim=96,hidden_dim=16):
    super().__init__()
    # self.batch_size = batch_size
    self.emb_dim = emb_dim
    self.num_steps = num_steps
    self.hidden_dim = hidden_dim
    self.sudoku_mask = sudoku_mask.reshape(64,64)

    # self.edges = sudoku_edges() 
    # batch_edges = torch.tensor([(i + (b * 64), j + (b * 64)) for b in range(self.batch_size) for i, j in self.edges])
    # n_edges = batch_edges.size()[0]

    # edge_features = torch.zeros((n_edges, 1))

   # positions = torch.tensor([[(i, j) for i in range(8) for j in range(8)] for b in range(self.batch_size)])

    self.embed_input = nn.Embedding(9,emb_dim)
    self.embed_rows = nn.Embedding(8,emb_dim)
    self.embed_cols = nn.Embedding(8,emb_dim)



    self.MLP = nn.Sequential(nn.Linear(3*emb_dim,mlp_dim),
                            #  nn.BatchNorm1d(hidden_dim), # for every one
                             nn.ReLU(),
                             nn.Linear(mlp_dim,mlp_dim),
                            #  nn.BatchNorm1d(hidden_dim), 
                             nn.ReLU(),
                             nn.Linear(mlp_dim,mlp_dim),
                            #  nn.BatchNorm1d(hidden_dim), 
                             nn.ReLU(),
                             nn.Linear(mlp_dim,hidden_dim))
    self.f = nn.Sequential(nn.Linear(2*hidden_dim,mlp_dim),
                            #  nn.BatchNorm1d(hidden_dim), # for every one
                             nn.ReLU(),
                             nn.Linear(mlp_dim,mlp_dim),
                            #  nn.BatchNorm1d(hidden_dim), 
                             nn.ReLU(),
                             nn.Linear(mlp_dim,mlp_dim),
                            #  nn.BatchNorm1d(hidden_dim), 
                             nn.ReLU(),
                             nn.Linear(mlp_dim,hidden_dim))
    
    self.MLP2 = nn.Sequential(nn.Linear(2*hidden_dim,mlp_dim),
                            #  nn.BatchNorm1d(hidden_dim), # for every one
                             nn.ReLU(),
                             nn.Linear(mlp_dim,mlp_dim),
                            #  nn.BatchNorm1d(hidden_dim), 
                             nn.ReLU(),
                             nn.Linear(mlp_dim,mlp_dim),
                            #  nn.BatchNorm1d(hidden_dim), 
                             nn.ReLU(),
                             nn.Linear(mlp_dim,hidden_dim))

    self.lstm_cell = nn.LSTMCell(input_size = hidden_dim, hidden_size = hidden_dim)
    self.rows = torch.Tensor([i for _ in range(8) for i in range(8)])
    self.cols = torch.Tensor([i for i in range(8) for _ in range(8)])

    # zero initalise the state?
    # self.state = (torch.zeros((self.batch_size,self.n_hidden)), torch.zeros((self.batch_size, self.n_hidden)))
    self.output = nn.Linear(in_features = hidden_dim, out_features = 8) #8?

  def forward(self,sudoku,answers):
    bsz,_ = sudoku.shape
    device = sudoku.device
    #sudoku--> bsz,64
    rows = self.rows.to(device)
    columns = self.cols.to(device)
    emb_sudoku = self.embed_input(sudoku)
    # print(emb_sudoku)
    emb_row = self.embed_rows(rows.unsqueeze(0).long())
    emb_col = self.embed_cols(columns.unsqueeze(0).long())
    # rows 1,64--> 1,64,embed_dim
    x = torch.cat([emb_sudoku,emb_row.repeat(bsz,1,1),emb_col.repeat(bsz,1,1)],dim=2)
    x = self.MLP(x)

    # a[0][0]--> 0th sudoku 0th pos embedding repeated 64 times
    # b[0][0]--> 0th sudoku all pos embeddings 

    s = torch.zeros(bsz*64,self.hidden_dim).to(device) # (h0,c0) of lstm 
    h = x
    x = x.view(bsz*64,-1)
    loss = 0
    # outputs = []
    # log_losses = []
    for steps in range(self.num_steps):
      # print(h)
      copy_every_element = h.unsqueeze(2).repeat(1,1,64,1)
      copy_every_sudoku = h.unsqueeze(1).repeat(1,64,1,1) 
      all_possible_messages = torch.cat([copy_every_element,copy_every_sudoku],dim=3)
      messages = all_possible_messages[:,self.sudoku_mask,:]
      # print(messages.shape)
      # ss = time.time()
      messages = self.f(messages)
      # ee = time.time()
      # print(ee-ss)
      new = torch.zeros(bsz,64,64,self.hidden_dim).to(device)
      new[:,self.sudoku_mask,:] = messages
      m = new.sum(dim=2) #m--> bsz,64,hidden_dim
      m = m.view(bsz*64,-1)
      lstm_inp = self.MLP2(torch.cat([x,m],dim=1)) #bsz*64,2*hidden_dim
      h = h.view(bsz*64,-1)
      h,s = self.lstm_cell(lstm_inp,(h,s))
      # print(s)
      h = h.view(bsz,64,-1)
      out = self.output(h)
      # print(out)
      # outputs.append(out)
      # log_losses.append(F.cross_entropy(out.reshape(bsz*64,8),answers.reshape(bsz*64)-1))
      l = nn.CrossEntropyLoss()(out.reshape(bsz*64,8),answers.reshape(bsz*64)-1)
      loss+=l
      # assert 1==2
      # print(l)
    return out,loss/(1.0*self.num_steps)

net = RRN().to(device)
optimizer = torch.optim.Adam(net.parameters(),weight_decay=1e-5)
# loss_fn = nn.CrossEntropyLoss()
def train_model(model,optimizer,num_epochs=100,patience=100):
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
          
          activations=[]

          # Each epoch has a training and validation phase
          for phase in ['train', 'val','test']:
              if phase == 'train':
                  model.train()  # Set model to training mode
              else:
                  model.eval()   # Set model to evaluate mode

              running_loss = 0.0
              running_corrects = 0 
              running_total = 0
              num_batches=0
              digit_corrects = 0
              digit_total = 0
              # predlist=[]
              # labellist=[]

  
              # Iterate over data.
              # for inputs, labels in dataloaders[phase]:
              for i,(sudoku,solved) in tqdm(enumerate(dataloaders[phase])):
        
                  su = sudoku.to(device)
                  so = solved.to(device)
                  
                  # forward
                  # track history if only in train
                  with torch.set_grad_enabled(phase == 'train'):
                      logits,loss = model(su,so)
                      _, preds = torch.max(logits, 2)
                      # los
                      # loss = loss_fn(logits, labels)


                      # print(inputs.shape)
                      


                      # backward + optimize only if in training phase
                      if phase == 'train':
                          optimizer.zero_grad()
                          loss.backward()
                          optimizer.step()
                      

                  # statistics
    
                  running_loss += loss.item()
                  digit_corrects += ((preds+1)==so.data).sum()
                  digit_total += len(so)*64
                  corrects = ((preds+1)==so.data).sum(dim=1) == 64
                  running_corrects += corrects.sum()
                  running_total += len(solved)
                  num_batches+=1
              # if phase == 'train':
                  # scheduler.step()

              epoch_loss = running_loss/num_batches
              epoch_acc = running_corrects/running_total
              digit_acc = digit_corrects/digit_total

              print('{} Loss: {:.4f} sudoku acc: {:.4f} digit acc: {:.4f}'.format(
                  phase, epoch_loss, epoch_acc,digit_acc))
              # scheduler.step()


              # deep copy the model
              if phase == 'val' and epoch_loss < best_loss:
                  print('BEST MODEL FOUND!!!')
                  clock=1
                  best_loss = epoch_loss
                  best_model_wts = copy.deepcopy(model.state_dict())
              # if phase == 'val' and epoch_acc > best_acc:
              #     print('BEST MODEL FOUND!!!')
              #     clock=1
              #     best_acc = epoch_acc
              #     best_model_wts = copy.deepcopy(model.state_dict())
        else:
          print('EARLY STOPPING!!!!!')
          break

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    # print('Best val acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

net = train_model(net,optimizer)

query_soln=[]
path_list = sorted(os.listdir(args.test_query))
for i,sudoku in tqdm(enumerate(test_query_sudoku)):
  s=sudoku.reshape(64)
  s=s.unsqueeze(0).to(device)
  answer = torch.ones(s.shape).to(device).long()
  logits,_ =net(s,answer)
  _, preds = torch.max(logits, 2)
  preds = preds.detach().cpu().reshape(64)+1
  query_soln.append([path_list[i]]+preds.tolist())

import csv
f = open(args.output_csv,'w')
write = csv.writer(f)
write.writerows(query_soln)

