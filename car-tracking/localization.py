# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 17:06:03 2018

@author: Ruijie Ni
"""

# =============================================================================
# We used a pre-trained ResNet-18 to train a localization model.
# Training set is from annotation.py
# Our work are based on PyTorch.
# The model is trained on a GeForce GTX 950M.
# =============================================================================


# %% The imports

import pickle
#import random
from time import time, sleep
import os

import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import sampler

import torchvision
import torchvision.transforms as T

from PIL import Image

import ultility


# %% Basic settings

# changed
#logdir = 'model'
logdir = 'car-tracking/model'
batch_size = 32
learning_rate = 1e-4
weight_decay = 5e-4
num_epochs = 300
decay_epochs = []
num_train = 3800
num_total = 3980

# unchanged
ttype = torch.cuda.FloatTensor # use GPU
dtype = torch.float32
device = torch.device('cuda')   
data_dir = './board-images-new'
anna_name = 'annotations'
input_size = 224
width = 640
height = 480

def parameterize(coords, w, h):
    return coords
    base_anchor = np.array(((w,h), (w,7*h), (7*w,7*h), (7*w,h))) / 8
    base_size = np.array((w, h)) * 3 / 4
    return (coords - base_anchor) / base_size

def inv_parameterize(t, w, h):
    return t
    base_anchor = np.array(((w,h), (w,7*h), (7*w,7*h), (7*w,h))) / 8
    base_size = np.array((w, h)) * 3 / 4
    return t * base_size + base_anchor


# %% Define the Dataset

class BoardLocalization(Dataset):
    def __init__(self, root, anna_file, transform=None):
        self.root = root
        self.annas = None
        self.transform = transform
        with open(os.path.join(root, anna_file), 'rb') as fo:
            self.annas = pickle.load(fo, encoding='bytes')
        
    def __getitem__(self, idx):
        anna = self.annas[idx]
        img = self.transform(Image.open(os.path.join(self.root, anna['file_name'])))
        coords = anna['coords']
        coords = np.array(coords).reshape((4,2))
        coords = parameterize(coords, width, height)
        coords = coords.reshape(8)
        return img, coords
    
    def __len__(self):
        return len(self.annas)


# %% Create 'Dataset's and 'DataLoader's

transform = T.Compose([
    T.Resize((input_size, input_size)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    
if __name__ == '__main__':    
    dset_train = BoardLocalization(data_dir, anna_name, transform=transform)
    loader_train = DataLoader(dset_train, batch_size=batch_size,
                              sampler=sampler.SubsetRandomSampler(range(num_train)))
    
    dset_val = BoardLocalization(data_dir, anna_name, transform=transform)
    loader_val = DataLoader(dset_val, batch_size=batch_size,
                            sampler=sampler.SubsetRandomSampler(range(num_train, num_total)))


# %% Utility functions

def smooth_L1(x, dim=0):
    """
    Inputs:
        - x: Tensor of size 4xN (by default) or size Nx4 (when dim=1)
    Returns:
        - loss: Tensor of size N
    """
    mask = (torch.abs(x) < 1).float()
    loss = torch.sum(mask*0.5*torch.pow(x, 2) + (1-mask)*(torch.abs(x)-0.5), dim)
    return loss


def check_acc(model, loader, total_batches=0):
    """Check accuracy for a specific Dataset by its DataLoader."""
    num_samples = num_correct = num_batches = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)
            scores = model(x)
            
            num_correct += torch.sum(torch.abs(scores - y) <= 40)
            num_samples += scores.shape[0] * scores.shape[1]
            
            num_batches += 1
            if num_batches == total_batches:
                break
            
        acc = float(num_correct) / num_samples
        return acc
    

# %% Evaluating procedure

def predict(image, debug=False):
    w, h  = image.shape[1],image.shape[0]
    edges = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.medianBlur(edges, 3)  # median blur to avoid dust near a corner
    
    image = Image.fromarray(np.uint8(image[:,:,::-1]))
    x = torch.unsqueeze(transform(image), 0)
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        scores = model(x)
        positions = np.reshape(scores.cpu().numpy(), (4,2))
        positions = inv_parameterize(positions, w, h)
        positions = positions.astype(np.int16)
        if debug:
            print('corners detected:')
            print(positions)
            
            print('corners refined offsets:')
            corner_images = []
            corner_pos = []
            
        for i in range(4):
            s = 20
            corner = cv.goodFeaturesToTrack(
                edges[max(positions[i,1]-s, 0) : min(positions[i,1]+s, h),
                      max(positions[i,0]-s, 0) : min(positions[i,0]+s, w)],
                1, 5e-2, 0
            )
            if debug:
                corner_images.append(
                    edges[max(positions[i,1]-s, 0) : min(positions[i,1]+s, h),
                          max(positions[i,0]-s, 0) : min(positions[i,0]+s, w)]
                )
                corner_pos.append(corner[0].reshape(-1))
            if corner is not None:
                print(corner-s)
                positions[i] += np.int16(corner[0]).reshape(-1) - s
        
        if debug:
            fig = plt.Figure()
            ultility.show(
                np.concatenate(corner_images[:2], axis=0),
                np.concatenate(corner_images[:1:-1], axis=0)
            )
            corner_pos = np.array(corner_pos)
            corner_pos[2:,0] += 2*s
            corner_pos[1:3,1] += 2*s
            plt.scatter(corner_pos[:,0], corner_pos[:,1])
            plt.show()
            
        return positions


# %% Initializing with pretrained ResNet-18

def init(predict=True):
    """Initialize the model, epoch and step, loss and acc summary."""
    
    global model, epoch, step
    global loss_summary, acc_summary
    
    epoch = 0
    step = 0
    
    for cur, _, files in os.walk('./'):  # check if we have the logdir already
        if cur == './{}'.format(logdir):  # we've found it
            # load basic resnet18
            model = torchvision.models.resnet18(pretrained=False, num_classes=8)
            
            # find the latest checkpoint file (.pkl)
            prefix, suffix = 'fine-tune-', '.pkl'
            file = None
            for ckpt in files:
                if not ckpt.endswith(suffix): continue
                info = ckpt[ckpt.find(prefix)+len(prefix) : ckpt.rfind(suffix)]
                e, s= [int(i)+1 for i in info.split('-')]
                if e > epoch or e == epoch and s > step:
                    epoch = e
                    step = s
                    file = ckpt
            # load the parameters from the file
            if file:
                print('Recovering from {}/{}'.format(logdir, file))
                model.load_state_dict(torch.load('{}/{}'.format(logdir, file)))
            else:
                print('***ERROR*** No .pkl file was found!')
            
            if not predict:
                # open the summary file
                with open('{}/summary'.format(logdir), 'rb') as fo:
                    dic = pickle.load(fo, encoding='bytes')
                    loss_summary = dic['loss']
                    acc_summary = dic['acc']
                
            break
        
    else:  # there's not
        if predict:
            print('***ERROR*** {}/ not found!'.format(logdir))
            return
        os.mkdir(logdir)
        # load pretrained resnet18
        model = torchvision.models.resnet18(pretrained=True)
        # change the number of classes
        #children = [child for child in model.classifier.children()][:-1]
        #fc = torch.nn.Linear(4096, 8)
        #children.append(fc)
        #model.classifier = torch.nn.Sequential(*children)
        #nn.init.kaiming_normal_(fc.weight)
        model.fc = torch.nn.Linear(model.fc.in_features, 8)
        nn.init.kaiming_normal_(model.fc.weight)
        
        loss_summary = []
        acc_summary = []
        
    # move to GPU
    model = model.to(device=device)


# %% Save

def save_model(e, step):
    filename = '{}/fine-tune-{}-{}.pkl'.format(logdir, e, step)
    torch.save(model.state_dict(), filename)
    print('Saved model successfully')
    print('Next epoch will start 10s later')
    sleep(10)


def save_summary():
    file = open('{}/summary'.format(logdir), 'wb')
    pickle.dump({'loss':loss_summary, 'acc':acc_summary}, file)
    file.close()
    

# %% Training procedure

def train(optimizer, num_epochs, print_every=100):
    global model, epoch, step, learning_rate
    
    tic = time()
    
    for e in range(epoch, num_epochs):
        #print('- Epoch {}'.format(e))
        
        for x, y in loader_train:
            model.train()  # put model to train mode
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype)
            
            scores = model(x)
            loss = smooth_L1(scores - y)
            #loss = torch.pow(scores - y, 2).view(-1, 4, 2)
            loss = torch.sum(loss) / x.shape[0]
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            
            loss_summary.append((step, loss.item()))
            
            if step % print_every == 0:
                print('-- Iteration {it}, loss = {loss:.4f}'.format(
                        it=step,loss=loss.item()), end=', ')
                
                train_acc = check_acc(model, loader_train, total_batches=20)
                print('train accuracy = {:.2f}%'.format(100 * train_acc), end=', ')
                
                val_acc = check_acc(model, loader_val)
                print('val accuracy = {:.2f}%'.format(100 * val_acc))
                
                acc_summary.append((step, train_acc, val_acc))
                
                save_summary()
                
            step += 1
        
        # save model
        if e % 20 == 0:
            save_model(e, step)
        
        if e in decay_epochs:
            epoch = e+1
            learning_rate /= 10
            return False
        
    toc = time()
    print('Use time: {}s'.format(toc-tic))
    
    return True


# %% Plot the summary

def plot(tau=200):
    # plot accuracy
    smooth = weighted_linear_regression(acc_summary, tau)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plot val accracy
    plt.plot([pair[0] for pair in acc_summary],
             [pair[2] for pair in acc_summary],
             color='#054E9F', linewidth=3, alpha=0.25)
    plt.plot([pair[0] for pair in smooth],
             [pair[2] for pair in smooth],
             color='#054E9F', linewidth=3)
    # plot train accuracy
    plt.plot([pair[0] for pair in acc_summary],
             [pair[1] for pair in acc_summary],
             color='coral', linewidth=3, alpha=0.25)
    plt.plot([pair[0] for pair in smooth],
             [pair[1] for pair in smooth],
             color='coral', linewidth=3)
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.plot([-10000,100000], [0,0], linewidth=2, color='grey')
    plt.plot([0,0], [-1,2], linewidth=2, color='grey')
    plt.xlim(xlim)
    plt.ylim([ylim[0], 1])
    plt.grid()
    for axis in ['top','right']:
        ax.spines[axis].set_linewidth(0)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_color('grey')
    plt.savefig('{}/acc.png'.format(logdir))
    plt.show()
    
    # plot loss
    smooth = weighted_linear_regression(loss_summary, tau)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot([pair[0] for pair in loss_summary],
             [pair[1] for pair in loss_summary],
             color='coral', linewidth=3, alpha=0.25)
    plt.plot([pair[0] for pair in smooth],
             [pair[1] for pair in smooth],
             color='coral', linewidth=3)
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.plot([-10000,100000], [0,0], linewidth=2, color='grey')
    plt.plot([0,0], [-1,20], linewidth=2, color='grey')
    plt.xlim(xlim)
    plt.ylim([ylim[0], int(ylim[1])])
    plt.grid()
    for axis in ['top','right']:
        ax.spines[axis].set_linewidth(0)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_color('grey')
    plt.savefig('{}/loss.png'.format(logdir))
    plt.show()


def weighted_linear_regression(summary, tau):
    smooth = [[pair[0]] for pair in summary]
    stretch = 64
    
    mat = np.array(summary)
    n = mat.shape[0]
    x, Y = mat[:,0:1], mat[:,1:mat.shape[1]]
    X = np.hstack((np.ones((n,1)), x))
    
    for j in range(Y.shape[1]):
        y = Y[:, j:j+1]
        for i in range(n):
            lo, hi = i-stretch, i+stretch
            if lo < 0: lo = 0
            if hi > n: hi = n
            W = np.diagflat(np.exp(-np.square(x[lo:hi]-x[i]) / (2*tau**2)))
            theta = np.dot(np.linalg.inv(X[lo:hi,:].T.dot(W).dot(X[lo:hi,:])),
                           (X[lo:hi,:].T.dot(W).dot(y[lo:hi,:])))
            smooth[i].append(float(theta[0]+x[i]*theta[1]))
    
    return smooth


# %% Test
    
def test():
    src = ['board-images-new/image{}.jpg'.format(i) for i in range(180, 200)]
    for img in src:
        plt.cla()
        image = cv.imread(img)
        ultility.show(image)
        positions = predict(image)
        plt.scatter(positions[:,0], positions[:,1])
        plt.pause(1)


# %% Main (train neural net)

def main():
    while(True):
        init()
        
        params = [{'params': m.parameters()} for m in model.children()]
        params[-1]['lr'] = learning_rate * 10
        if train(optim.SGD(params, lr=learning_rate,
                           momentum=0.9, weight_decay=weight_decay),
                num_epochs=num_epochs):
            break
    
    #plot()


if __name__ == '__main__':
    main()
