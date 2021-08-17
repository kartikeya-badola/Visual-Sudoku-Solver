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
import random

parser = argparse.ArgumentParser(description='COL 870 Assignment 2.1')
parser.add_argument('--train_path', default='/content/Assignment 2/visual_sudoku/train', type=str, help='path to train images')
parser.add_argument('--sample_images', default = '/content/Assignment 2/sample_images.npy', type=str, help='path to sample images.npy')
parser.add_argument('--gen_path', default='gen9k.npy', type=str, help='path to gen9k.npy')
parser.add_argument('--target_path', default='target9k.npy', type=str, help='path to target9k.npy')


args = parser.parse_args()

img_list = []
query_path = os.path.join(args.train_path,'query/')
for l in tqdm(os.listdir(query_path)):
  pil_img = Image.open(query_path+l)
  tensor = transforms.ToTensor()(pil_img)
  img_list.append(tensor.unsqueeze(0))



imgs = torch.cat(img_list,dim=0)
patches = imgs.unfold(2, 28, 28).unfold(3, 28, 28)
patches = patches.reshape(len(img_list),1,64,28,28).permute(0,2,1,3,4)
patches=1-patches
gt = torch.Tensor(1-np.load(args.sample_images)/255.0)

gt = gt[:-1]
gt = gt.unsqueeze(1).unsqueeze(0).repeat(64,1,1,1,1)
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

os.mkdir('new_clusters')
for i in range(9):
  os.mkdir('new_clusters/'+str(i))

final_imgs = []
final_labels = []
for j,imgs in tqdm(enumerate(patches)):
  imgs = imgs.unsqueeze(1).repeat(1,9,1,1,1)
  # print(imgs.shape,gt.shape)
  for i in range(len(imgs)):
    x=ssim(imgs[i],gt[i],data_range=1,size_average=False,nonnegative_ssim=True)
    # print(x)
    x = x/x.sum()
    # print(x)
    # x_sorted = x.sort()[0]
    # print(x_sorted)
    # print(x_sorted[-1]-x_sorted[-2])
    # if x_sorted[-1]-x_sorted[-2] >0.1:
    # print(x)
    # assert 1==2
    if x.max()>0.3:
      l = x.argmax()
    # if x.max()>0.:
      # final_imgs.append(imgs[i][0].unsqueeze(0))
      # final_labels.append(l.item())


    # print(l)
      torchvision.utils.save_image(imgs[i][0].unsqueeze(0),'new_clusters/'+str(l.item())+'/'+str(j)+'_'+str(i)+'T'+'.png')
  # break



# Repeat loop for targets

img_list = []
target_path = os.path.join(args.train_path,'target/')
for l in tqdm(os.listdir(target_path)):
  pil_img = Image.open(target_path+l)
  tensor = transforms.ToTensor()(pil_img)
  img_list.append(tensor.unsqueeze(0))


imgs = torch.cat(img_list,dim=0)
patches = imgs.unfold(2, 28, 28).unfold(3, 28, 28)
patches = patches.reshape(len(img_list),1,64,28,28).permute(0,2,1,3,4)
patches=1-patches
gt = torch.Tensor(1-np.load(args.sample_images)/255.0)

gt = gt[:-1]
gt = gt.unsqueeze(1).unsqueeze(0).repeat(64,1,1,1,1)
# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

# os.mkdir('new_clusters')
# for i in range(9):
#   os.mkdir('new_clusters/'+str(i))

final_imgs = []
final_labels = []
for j,imgs in tqdm(enumerate(patches)):
  imgs = imgs.unsqueeze(1).repeat(1,9,1,1,1)
  # print(imgs.shape,gt.shape)
  for i in range(len(imgs)):
    x=ssim(imgs[i],gt[i],data_range=1,size_average=False,nonnegative_ssim=True)
    # print(x)
    x = x/x.sum()
    # print(x)
    # x_sorted = x.sort()[0]
    # print(x_sorted)
    # print(x_sorted[-1]-x_sorted[-2])
    # if x_sorted[-1]-x_sorted[-2] >0.1:
    # print(x)
    # assert 1==2
    if x.max()>0.3:
      l = x.argmax()
    # if x.max()>0.:
      # final_imgs.append(imgs[i][0].unsqueeze(0))
      # final_labels.append(l.item())


    # print(l)
      torchvision.utils.save_image(imgs[i][0].unsqueeze(0),'new_clusters/'+str(l.item())+'/'+str(j)+'_'+str(i)+'.png')
  # break


minimum_images = 1000000

folders = ([name for name in os.listdir('new_clusters/')
            if os.path.isdir(os.path.join('new_clusters/', name))])
for folder in folders:
  contents = os.listdir(os.path.join('new_clusters/', folder))
  minimum_images = min(len(contents),minimum_images)

print('Minimum Images: ' + str(minimum_images))

for folder in folders:
  contents = os.listdir(os.path.join('new_clusters/', folder))
  print('Original Size: ' + str(len(contents)))
  #print(contents)
  random.shuffle(contents)
  to_be_deleted = contents[minimum_images:]
  for filename in contents:
    file_path = os.path.join(os.path.join('new_clusters/', folder), filename)
    try:
      if os.path.isfile(file_path) and filename in to_be_deleted:
        os.remove(file_path)
    except Exception as e:
      print('Failed to delete %s, reason: %s' %(file_path, e))

  print('New Size: ' + str(len(os.listdir(os.path.join('new_clusters/', folder)))))





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
transform = transforms.Compose([transforms.Grayscale(),
        transforms.ToTensor()])
dataset = datasets.ImageFolder('new_clusters',transform)
dataset_size = len(dataset)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=100,
                                             shuffle=True, num_workers=2)


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

loss = nn.BCELoss()
# loss = nn.MSELoss()
g = G(9,100).to(device)
d = D(9).to(device)
Goptim = optim.SGD(g.parameters(),lr=0.1,momentum=0.5)
Doptim = optim.SGD(d.parameters(),lr=0.1,momentum=0.5)
Gscheduler = optim.lr_scheduler.ExponentialLR(Goptim,1/1.00004)
Dscheduler = optim.lr_scheduler.ExponentialLR(Doptim,1/1.00004)
# Goptim = optim.Adam(g.parameters(),lr=0.0002,betas=(0.5,0.999))
# Doptim = optim.Adam(d.parameters(),lr=0.0002,betas=(0.5,0.999))

def sample_image(g,n_classes, hidden_dim,name):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z =  torch.FloatTensor(n_classes**2,hidden_dim).uniform_(0,1).to(device)
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_classes) for num in range(n_classes)])
    labels = torch.Tensor(labels).long().to(device)
    gen_imgs = g(z, labels).reshape(n_classes**2,1,28,28)
    torchvision.utils.save_image(gen_imgs.data, "generator_images/"+name+".png", nrow=n_classes, normalize=True)

os.mkdir('generator_images')

def train_model(g,d,Goptim,Doptim,Gscheduler,Dscheduler,n_classes,hidden_dim,num_epochs=100,patience=100):
# def train_model(g,d,Goptim,Doptim,n_classes,hidden_dim,num_epochs=100,patience=100):

    clock=0
    since = time.time()

    # best_D_wts = copy.deepcopy(D.state_dict())
    # best_G_wts = copy.deepcopy(G.state_dict())
    best_loss = 1000000
    momentum_factor = 0
    num_iters = 0
    for epoch in range(num_epochs):
        if(clock<patience):
          clock=clock+1
          glr=0
          gmom=0
          for pg in Goptim.param_groups:
              pg['momentum'] = pg['momentum']+momentum_factor
              glr=pg['lr']
              gmom=pg['momentum']
          dlr=0
          dmom=0
          for pg in Doptim.param_groups:
              pg['momentum'] = pg['momentum']+momentum_factor
              dlr=pg['lr']
              dmom=pg['momentum']
          print('G_lr:',glr,'G_momentum',gmom)
          print('D_lr:',dlr,'D_momentum',dmom)
          print('Epoch {}/{}'.format(epoch, num_epochs - 1))
          print('-' * 10)
          # alpha = 1/(num_epochs-epoch+1)**0.3
          # print('alpha is:',alpha)

          # Each epoch has a training and validation phas
          # g.train()
          # d.train()  # Set model to training mode

          running_loss = 0.0
          running_loss_G = 0.0
          running_loss_D = 0.0
          running_corrects = 0

          # Iterate over data.
          for i,(inputs, labels) in enumerate(tqdm(dataloader)):
            if num_iters<55000:
                num_iters+=1
                bsz = labels.shape[0]
                real = inputs.view(bsz,-1).to(device)
                labels = labels.to(device)
                noise = torch.FloatTensor(bsz,hidden_dim).uniform_(0,1).to(device)
                # noise = torch.normal(0,1,(bsz,hidden_dim)).to(device)
                one_label = torch.ones(bsz,1).to(device)
                zero_label = torch.zeros(bsz,1).to(device)
                # one_label = torch.autograd.Variable(torch.cuda.FloatTensor(bsz, 1).fill_(1.0), requires_grad=False)
                # zero_label = torch.autograd.Variable(torch.cuda.FloatTensor(bsz, 1).fill_(0.0), requires_grad=False)
                # noise = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (bsz,hidden_dim))))

                

            

                gen_labels = torch.randint(0,n_classes,[bsz]).to(device)
                Goptim.zero_grad()
                fake = g(noise,gen_labels)
                # d.eval()
                score = d(fake,gen_labels)
                # d.train()
                loss_G = loss(score,one_label)
                loss_G.backward()
                Goptim.step()
                Gscheduler.step()

                '''
                Training discriminator with 2 batches of input. one with fake images, one with real images
                '''
                  
                Doptim.zero_grad()
                real_scores = d(real,labels)
                pred_real = real_scores > 0.5
                loss_D_real =  loss(real_scores,one_label)

                fake_scores = d(fake.detach(),gen_labels)
                pred_fake = fake_scores > 0.5

                loss_D_fake = loss(fake_scores,zero_label)
                loss_D = 0.5*(loss_D_real+loss_D_fake)
                loss_D.backward()
                Doptim.step()
                Dscheduler.step()
                    # D_scheduler.step()
                pred = torch.cat([pred_real,pred_fake],dim=0)
                true = torch.cat([one_label,zero_label],dim=0)
                Loss = loss_G+loss_D
                if i%1000==0:
                  print('saving images')
                  sample_image(g,n_classes,hidden_dim,str(epoch)+'_'+str(i))


                running_loss += Loss.item() * inputs.size(0)
                running_loss_G += loss_G.item()*inputs.size(0)
                running_loss_D += loss_D.item()*inputs.size(0)
                running_corrects += 0.5*torch.sum(pred == true)
            # if phase == 'train':
                # scheduler.step()
            else:
              print("Training stopped")
              break

          epoch_loss = running_loss / dataset_size
          epoch_loss_G = running_loss_G / dataset_size
          epoch_loss_D = running_loss_D / dataset_size
          epoch_acc = running_corrects / dataset_size


          print('Loss: {:.4f} G_Loss: {:.4f} D_Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss,epoch_loss_G,epoch_loss_D, epoch_acc))
          torch.save(d,'D'+str(epoch)+'.pth')
          torch.save(g,'G'+str(epoch)+'.pth')


        
              

        # deep copy the model
        if epoch_loss < best_loss:
            print('BEST MODEL FOUND!!!')
            clock=1
            best_loss = epoch_loss
            torch.save(d,'best_loss_D.pth')
            torch.save(g,'best_loss_G.pth')
        elif clock==patience:
          print('EARLY STOPPING!!!!!')
          break

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best loss: {:4f}'.format(best_loss))

    return

train_model(g,d,Goptim,Doptim,Gscheduler,Dscheduler,n_classes=9,hidden_dim=100,num_epochs=5,patience=5)
g=torch.load('G1.pth').to(device)
g.eval()
z =  torch.FloatTensor(9000,100).uniform_(0,1).to(device)
# Get labels ranging from 0 to n_classes for n rows
labels = np.array([num for num in range(9) for _ in range(1000)])
labels = torch.Tensor(labels).long().to(device)
gen_imgs = g(z, labels).cpu().detach()

np.save(args.gen_path,gen_imgs.reshape(9000,1,28,28))

target_list = []
for i in range(9):
    folder = os.listdir('new_clusters/'+str(i))
    for j in range(1000):
        path = 'new_clusters/'+str(i)+'/'+folder[j]
        pil_img = Image.open(path)
        tensor = transforms.ToTensor()(pil_img)
        target_list.append(tensor.unsqueeze(0))

target_images = torch.cat(target_list,dim=0)
# target_images = 1-target_images
np.save(args.target_path,target_images)

os.mkdir('fid_0')
os.mkdir('fid_1')
for i,im in enumerate(gen_imgs.reshape(9000,1,28,28)):
  torchvision.utils.save_image(im.unsqueeze(0),'fid_1/'+str(i)+'.png')
for i,im in enumerate(target_images):
  torchvision.utils.save_image(im.unsqueeze(0),'fid_0/'+str(i)+'.png')
        
