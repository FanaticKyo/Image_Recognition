#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# In[2]:


torch.cuda.is_available()


# In[3]:


# def train(model, data_loader, test_loader, task='Classification'):
#     model.train()

#     for epoch in range(numEpochs):
#         avg_loss = 0.0
#         for batch_num, (feats, labels) in enumerate(data_loader):
#             feats, labels = feats.to(device), labels.to(device)
            
#             optimizer.zero_grad()
#             outputs = model(feats)[1]

#             loss = criterion(outputs, labels.long())
#             loss.backward()
#             optimizer.step()
            
#             avg_loss += loss.item()

#             if batch_num % 50 == 49:
#                 print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch+1, batch_num+1, avg_loss/50), end='\r')
#                 avg_loss = 0.0    
            
#             torch.cuda.empty_cache()
#             del feats
#             del labels
#             del loss
        
#         if task == 'Classification':
#             val_loss, val_acc = test_classify(model, test_loader)
# #             train_loss, train_acc = test_classify(model, data_loader)
# #             print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.
# #                   format(train_loss, train_acc, val_loss, val_acc))
#             print('Val Loss: {:.4f}\tVal Accuracy: {:.4f}'.format(val_loss, val_acc))
#         else:
#             test_verify(model, test_loader)


# def test_classify(model, test_loader):
#     model.eval()
#     test_loss = []
#     accuracy = 0
#     total = 0

#     for batch_num, (feats, labels) in enumerate(test_loader):
#         feats, labels = feats.to(device), labels.to(device)
#         outputs = model(feats)[1]
        
#         _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
#         pred_labels = pred_labels.view(-1)
        
#         loss = criterion(outputs, labels.long())
        
#         accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
#         total += len(labels)
#         test_loss.extend([loss.item()]*feats.size()[0])
#         del feats
#         del labels
        
#     torch.cuda.empty_cache()
#     model.train()
#     return np.mean(test_loss), accuracy/total


# def test_verify(model, test_loader):
#     raise NotImplementedError


# In[4]:


numEpochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[5]:


train_tfms = transforms.Compose([
            transforms.ToTensor() 
           # don't use transforms.Normalize() for the first time
        ])


# In[6]:


train_dataset = torchvision.datasets.ImageFolder(root='train_data/large/', 
                                                 transform=train_tfms)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, 
                                               shuffle=True, num_workers=6)


# In[7]:


val_tfms = transforms.Compose([
            transforms.ToTensor()
        ])


# In[8]:


dev_dataset = torchvision.datasets.ImageFolder(root='validation_classification/medium/', 
                                               transform=val_tfms)
dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=256, 
                                             shuffle=False, num_workers=1)


# In[17]:


train_mean = 0
count = 0
for data, label in train_dataloader:
    train_mean += data.mean(axis=(0, 2, 3))
    count += 1
train_mean /= count


# In[42]:


train_std = 0
for data, label in train_dataloader:
    train_std += data.std(axis = (0, 2, 3))
train_std /= count


# In[44]:


val_mean = 0
count = 0
for data, label in dev_dataloader:
    val_mean += data.mean(axis=(0, 2, 3))
    count += 1
val_mean /= count


# In[46]:


val_std = 0
for data, label in dev_dataloader:
    val_std += data.std(axis = (0, 2, 3))
val_std /= count


# In[12]:


# train_mean = torch.tensor([0.3564, 0.2540, 0.3642])


# In[13]:


# train_std = torch.tensor([0.2171, 0.1618, 0.2238])


# In[14]:


# val_mean = torch.tensor([0.3558, 0.2537, 0.3636])
# val_std = torch.tensor([0.2169, 0.1617, 0.2236])


# In[15]:


train_tfms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std)
        ])
val_tfms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(val_mean, val_std)
        ])


# In[16]:


train_dataset = torchvision.datasets.ImageFolder(root='train_data/medium/', 
                                                 transform=train_tfms)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, 
                                               shuffle=True, num_workers=1)
dev_dataset = torchvision.datasets.ImageFolder(root='validation_classification/medium/', 
                                               transform=val_tfms)
dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=256, 
                                             shuffle=False, num_workers=1)


# In[17]:


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + x, inplace=True)
        return out


# In[19]:


# class Network(nn.Module):
#     def __init__(self, feat_dim=2):
#         super(Network, self).__init__()
        
#         self.layers = []
        
#         self.layers.append(nn.Conv2d(3, 64, 3, 2, 1, bias=False))
#         self.layers.append(nn.BatchNorm2d(64))
#         self.layers.append(nn.ReLU(inplace=True))
# #         self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False))
#         self.layers.append(ConvBlock(64, 256, 2))
    
#         for i in range(2):
#             self.layers.append(ConvBlock(256, 256, 1))
            
#         self.layers.append(ConvBlock(256, 512, 2))
        
#         for j in range(3):
#             self.layers.append(ConvBlock(512, 512, 1))
        
#         self.layers.append(ConvBlock(512, 1024, 2))
        
#         for k in range(5):
#             self.layers.append(ConvBlock(1024, 1024, 1))
        
#         self.layers.append(ConvBlock(1024, 2048, 2))
        
#         for j in range(2):
#             self.layers.append(ConvBlock(2048, 2048, 1))
            
#         self.layers = nn.Sequential(*self.layers)
#         self.linear_label = nn.Linear(2048, 2300, bias=True)
        
#         # For creating the embedding to be passed into the Center Loss criterion
#         self.linear_closs = nn.Linear(2048, feat_dim, bias=False)
#         self.relu_closs = nn.ReLU(inplace=True)
    
#     def forward(self, x, evalMode=False):
#         output = x
#         output = self.layers(output)
            
#         output = F.avg_pool2d(output, [output.size(2), output.size(3)], stride=1)
#         output = output.reshape(output.shape[0], output.shape[1])
#         self.embedding = output
#         label_output = self.linear_label(output)
#         label_output = label_output/torch.norm(self.linear_label.weight, dim=1)
        
#         # Create the feature embedding for the Center Loss
#         closs_output = self.linear_closs(output)
#         closs_output = self.relu_closs(closs_output)

#         return closs_output, label_output


# In[20]:


class Network(nn.Module):
    def __init__(self, feat_dim=2):
        super(Network, self).__init__()
        
        self.layers = []
        
        self.layers.append(nn.Conv2d(3, 512, 3, 2, 0, bias=False))
        self.layers.append(nn.BatchNorm2d(512))
        self.layers.append(nn.ReLU(inplace=True))   
#         self.layers.append(BasicBlock(64, 64, 1))
#         self.layers.append(BasicBlock(64, 64, 1))
        
#         self.layers.append(nn.Conv2d(64, 256, 3, 2, 1, bias=False))
#         self.layers.append(nn.BatchNorm2d(256))
#         self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(BasicBlock(512, 512, 1))
        self.layers.append(BasicBlock(512, 512, 1))
        self.layers.append(BasicBlock(512, 512, 1))
        
        self.layers.append(nn.Conv2d(512, 1024, 3, 2, 0, bias=False))
        self.layers.append(nn.BatchNorm2d(1024))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(BasicBlock(1024, 1024, 1))
        self.layers.append(BasicBlock(1024, 1024, 1))
#         self.layers.append(BasicBlock(1024, 1024, 1))
        
#         self.layers.append(nn.Conv2d(1024, 2048, 3, 2, 0, bias=False))
#         self.layers.append(nn.BatchNorm2d(2048))
#         self.layers.append(nn.ReLU(inplace=True))
#         self.layers.append(BasicBlock(2048, 2048, 1))
#         self.layers.append(BasicBlock(2048, 2048, 1))


#         self.layers.append(nn.Conv2d(1024, 2048, 3, 2, 1, bias=False))
#         self.layers.append(nn.BatchNorm2d(2048))
#         self.layers.append(nn.ReLU(inplace=True))
#         self.layers.append(BasicBlock(2048, 2048, 1))
#         self.layers.append(BasicBlock(2048, 2048, 1))
        self.dropout = nn.Dropout2d(0.5)
        self.layers = nn.Sequential(*self.layers)
        self.linear_label = nn.Linear(1024, 2300, bias=False)
        
        # For creating the embedding to be passed into the Center Loss criterion
        self.linear_closs = nn.Linear(1024, feat_dim, bias=False)
        self.relu_closs = nn.ReLU(inplace=True)
    
    def forward(self, x, evalMode=False):
        output = x
        output = self.layers(output)
        output = self.dropout(output)
        output = F.avg_pool2d(output, [output.size(2), output.size(3)], stride=1)
        output = output.reshape(output.shape[0], output.shape[1])
        embedding = output
        label_output = self.linear_label(output)
        label_output = label_output/torch.norm(self.linear_label.weight, dim=1)
        
        # Create the feature embedding for the Center Loss
        closs_output = self.linear_closs(output)
        closs_output = self.relu_closs(closs_output)

        return closs_output, label_output, embedding


# In[21]:


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight.data)
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)


# In[22]:


import time


# In[23]:


def train_closs(model, data_loader, test_loader, task='Classification'):
    model.train()
    
    for epoch in range(numEpochs):
        start = time.time()
        avg_loss = 0.0
        for batch_num, (feats, labels) in enumerate(data_loader):
            feats, labels = feats.to(device), labels.to(device)
            
            optimizer_label.zero_grad()
            optimizer_closs.zero_grad()
            
            feature, outputs, embedding = model(feats)
            l_loss = criterion_label(outputs, labels.long())
            c_loss = criterion_closs(feature, labels.long())
            loss = l_loss + closs_weight * c_loss
            
            loss.backward()
            
            optimizer_label.step()
            # by doing so, weight_cent would not impact on the learning of centers
            for param in criterion_closs.parameters():
                param.grad.data *= (1. / closs_weight)
            optimizer_closs.step()
            
            avg_loss += loss.item()

            print('Epoch: {}   Batch: {} / {}    Avg-Loss: {:.4f}'.format(epoch+1, batch_num+1, len(train_dataloader), avg_loss), end='\r')
            avg_loss = 0.0    
            
            torch.cuda.empty_cache()
            del feats
            del labels
            del loss
        end = time.time()
        if task == 'Classification':
            val_loss, val_acc = test_classify_closs(model, test_loader)
#             train_loss, train_acc = test_classify_closs(model, data_loader)
            print('\nVal Loss: {:.4f}\tVal Accuracy: {:.4f}\tTime: {:.2f} s'.
                  format(val_loss, val_acc, end-start))
            torch.save(network, 'mynet.pth')
        else:
            test_verify(model, test_loader)


def test_classify_closs(model, test_loader):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        feature, outputs, embedding = model(feats)
        
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        
        l_loss = criterion_label(outputs, labels.long())
        c_loss = criterion_closs(feature, labels.long())
        loss = l_loss + closs_weight * c_loss
        
        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()]*feats.size()[0])
        del feats
        del labels

    model.train()
    return np.mean(test_loss), accuracy/total


# In[24]:


class CenterLoss(nn.Module):
    """
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes, feat_dim, device=torch.device('cpu')):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) +                   torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss


# In[25]:


network = Network(2)
network.apply(init_weights)


# In[26]:


device = 'cuda'


# In[27]:


closs_weight = 0.003
# network = torch.load('mynet.pth')

criterion_label = nn.CrossEntropyLoss()
criterion_closs = CenterLoss(2300, 2, device)
optimizer_label = torch.optim.Adam(network.parameters(), lr=0.0005)
optimizer_closs = torch.optim.SGD(criterion_closs.parameters(), lr=0.5)


# In[28]:


# torch.save(network, 'mynet.pth')


# In[29]:


network.train()
network.to(device)


# In[31]:


train_closs(network, train_dataloader, dev_dataloader)


# In[30]:


optimizer_label = torch.optim.Adam(network.parameters(), lr=0.00001)


# In[31]:


train_closs(network, train_dataloader, dev_dataloader)


# In[34]:


# For Fine Tuning


# In[35]:


# For Testing


# In[49]:


# test_mean = torch.tensor([0.3558, 0.2536, 0.3639])
# test_std = torch.tensor([0.2231, 0.1682, 0.2299])


# In[52]:


file_order = []
with open('test_order_classification.txt') as f:
    for line in f:
        file_order.append(line.strip())


# In[53]:


img_list = []


# In[54]:


for file in file_order:
    img_list.append('test_classification/medium/' + file)


# In[55]:


class ImageDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        img = test_tfms(img)
#         img = torch.transforms.ToTensor()(img)
        return img


# In[56]:


testset = ImageDataset(img_list)


# In[57]:


test_tfms = transforms.Compose([
            transforms.ToTensor()
        ])


# In[58]:


test_dataloader = DataLoader(testset, batch_size=512, shuffle=False, num_workers=1)


# In[60]:


test_mean = 0
count = 0
for data in test_dataloader:
    test_mean += data.mean(axis=(0, 2, 3))
    count += 1
test_mean /= count


# In[62]:


test_std = 0
for data in test_dataloader:
    test_std += data.std(axis = (0, 2, 3))
test_std /= count


# In[29]:


test_tfms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(test_mean, test_std)
        ])


# In[35]:


test_dataloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=1)


# In[36]:


# test_mean, test_std = online_mean_and_sd(test_dataloader)


# In[37]:


# print(test_mean, test_std)


# In[38]:


network.eval()
test_preds = torch.LongTensor()
for x_batch in test_dataloader:
    x_batch = x_batch.cuda()
    outputs = network(x_batch)[1]
    _, pred = torch.max(F.softmax(outputs, dim=1), 1)
    pred = pred.view(-1).to('cpu')
    test_preds = torch.cat((test_preds, pred), dim=0)


# In[39]:


t = test_preds.numpy()


# In[40]:


inv_map = {v: k for k, v in train_dataset.class_to_idx.items()}


# In[41]:


for i in range(len(t)):
    t[i] = inv_map[t[i]]


# In[42]:


t


# In[43]:


id = []
for file in file_order:
    id.append(file.split('.')[0])


# In[44]:


import pandas as pd
result = pd.DataFrame()
result['id'] = np.asarray(id)
result['label'] = t


# In[45]:


result


# In[46]:


result.to_csv('shengxu_classification.csv', index=False)


# In[ ]:


# Verification


# In[ ]:


validation_varification = []
with open('test_trials_verification_student.txt') as f:
    for line in f:
        validation_varification.append(line.split())


# In[ ]:


img0 = ['test_verification/' + i[0] for i in validation_varification]


# In[ ]:


img1 = ['test_verification/' + i[1] for i in validation_varification]


# In[ ]:


def get_test_mean_std(testdata):
    
    count = 0
    mean = torch.empty(3)
    p = torch.empty(3)

    for data in testdata:

        batch_size, color_channel, height, width = data.shape
        total = batch_size * height * width
        total_images = torch.sum(data, dim=[0, 2, 3])
        total_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (count * fst_moment + total_images) / (count + total)
        snd_moment = (count * snd_moment + total_square) / (count + total)
        count += total

    return mean, torch.sqrt(p - mean ** 2)


# In[ ]:


veri_tfms = transforms.Compose([
            transforms.ToTensor()
        ])


# In[ ]:


class VeriImageDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        img = veri_tfms(img)
#         img = torch.transforms.ToTensor()(img)
        return img


# In[ ]:


veri_set0 = VeriImageDataset(img0)


# In[ ]:


veri_dataloader0 = DataLoader(veri_set0, batch_size=512, shuffle=False, num_workers=0)


# In[ ]:


# veri_mean = torch.tensor([0.3559, 0.2538, 0.3639])


# In[ ]:


# veri_std = torch.tensor([0.2232, 0.1687, 0.2301])


# In[ ]:


veri_mean, veri_std = get_test_mean_std(veri_dataloader0)


# In[ ]:


veri_tfms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(veri_mean, veri_std)
        ])


# In[ ]:


veri_set0 = VeriImageDataset(img0)


# In[ ]:


veri_dataloader0 = DataLoader(veri_set0, batch_size=512, shuffle=False, num_workers=0)


# In[ ]:


veri_set1 = VeriImageDataset(img1)


# In[ ]:


veri_dataloader1 = DataLoader(veri_set1, batch_size=512, shuffle=False, num_workers=0)


# In[ ]:


network.eval()


# In[ ]:


veri0 = []
for x_batch in veri_dataloader0:
    x_batch = x_batch.cuda()
    outputs = network(x_batch)[2].cpu().detach().numpy()
    veri0.append(outputs)


# In[ ]:


veri1 = []
for x_batch in veri_dataloader1:
    x_batch = x_batch.cuda()
    outputs = network(x_batch)[2].cpu().detach().numpy()
    veri1.append(outputs)


# In[ ]:


def cos_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# In[ ]:


r = np.zeros((899965, ))
i = 0
for j in range(len(veri0)):
    for k in range(len(veri0[j])):
        r[i] = cos_similarity(veri0[j][k], veri1[j][k])
        i += 1


# In[ ]:


img = [i[0] + ' ' + i[1] for i in validation_varification]


# In[ ]:


import pandas as pd
result = pd.DataFrame()
result['trial'] = np.asarray(img)
result['score'] = r


# In[ ]:


result


# In[ ]:


result.to_csv('shengxu_verification.csv', index=False)

