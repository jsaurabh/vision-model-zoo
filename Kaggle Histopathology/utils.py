import os

import torch
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from fastai.vision import cnn_learner
from fastai.metrics import accuracy
from fastai.train import *
from sklearn.metrics import roc_auc_score

def read(img):
    img = cv.imread(img)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img

def subplot(trainDir, positive, negative):
    """
    https://www.kaggle.com/qitvision/a-complete-ml-pipeline-fast-ai
    """
    fig, ax = plt.subplots(2,5, figsize=(20,8))
    fig.suptitle('Scans of lymph nodes sections',fontsize=20)
    
    for idx, p in enumerate(negative['id'][:5]):
        path = os.path.join(trainDir, p)
        ax[0,idx].imshow(read(path + '.tif'))
        box = patches.Rectangle((32,32),32,32,linewidth=4,edgecolor='b',facecolor='none', linestyle=':', capstyle='round')
        ax[0,idx].add_patch(box)
    ax[0,0].set_ylabel('Negative samples', size='large')

    for idx, p in enumerate(positive['id'][:5]):
        path = os.path.join(trainDir, p)
        ax[1,idx].imshow(read(path + '.tif'))
        box = patches.Rectangle((32,32),32,32,linewidth=4,edgecolor='b',facecolor='none', linestyle=':', capstyle='round')
        ax[1,idx].add_patch(box)
    ax[1,0].set_ylabel('Positive samples', size='large')

def get_learner(dataBunch, architecture, metrics = accuracy, pretrained = True, callbacks = ShowGraph):
    return cnn_learner(data = dataBunch, base_arch = architecture, pretrained= True, metrics = accuracy, callback_fns = callbacks)

def auc_score(y_pred,y_true):
    score=roc_auc_score(y_true,torch.sigmoid(y_pred)[:,1])
    return score