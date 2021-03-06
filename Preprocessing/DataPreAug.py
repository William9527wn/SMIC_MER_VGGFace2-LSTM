#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Created on Monday May 25 10:51 2020
Used to pre-process dataset 
@author: Keira - github.com/Keira. Bai
a.function to divide data set into 5 groups for cross-validation, return a 5*N dimention matrix 
b.function to extract training data into fixed size sequences, sequences consist by consecuous internal
c.function to extract validation data, sequences consist by raw image and adding frames to the same size
d.function to read image
"""
#Data preprocessing version 2.0
#1. Random steps between frames within a slide window;
#2. Fixed-size of slide window(Can't be implemented on seleframe != 11);
#3. No flipping
#4. Raw data for validation


# In[2]:



import PIL
import torch
import glob as gb
import numpy as np
from torch.utils import data
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
plt.ion()   # interactive mode


# In[ ]:


def CrossAllocation(datapaths,k):
    neg_dir = "*/negative/*"
    pos_dir = "*/positive/*"
    sur_dir = "*/surprise/*"
    non_dir = "non_micro/*"
    imgpath = "/*.bmp"
    sequ_list = []
    label_list = []
    Val_tra_tes_list = []
    label = 0
    #   Load in all folders adding expression
    for vid_dir in [neg_dir, pos_dir, sur_dir, non_dir]:        
        sequ = gb.glob(datapaths+vid_dir)
        sequ_list += sequ
        label_list += [label for i in range(len(sequ))]
        label += 1
    #Devide All Data into 5 groups
    for i in range(k):
        if k-i>1:
            train_list, valid_list, train_label, valid_label = train_test_split(sequ_list, label_list,                                           test_size=1/(k-i), random_state=42)
            Val_tra_tes_list.append([valid_list, valid_label])
            
        else:
            Val_tra_tes_list.append([train_list, train_label])

        sequ_list = train_list
        label_list = train_label   

    return Val_tra_tes_list 


# In[10]:


def InputImagewithSlide(tra_tes_list, seleframe = 11):    
    imgpath = "/*.bmp"
    sequ_list = []
    
    exp_list = []
    step_list = [1,2,3,4,5,6]
    expCount = [0,0,0,0,0,0,0,0]

    for group in tra_tes_list:    
        for folder, exp in zip(group[0], group[1]):
            img_list = sorted(gb.glob(folder + imgpath))
            img_len = len(img_list)
            expCount[exp] += img_len
            #control step to keep the amout balance between expressions
            if exp == 3:
                step = 3
            else:
                step = 1
            for i in range(0, img_len-seleframe+step, step):
                    if ((img_len-i) >= (seleframe+step-1)):#sequence is enough for a seleframe+step
                        for s in step_list:#assign inner step for frame
                            if img_len-i >= seleframe*s:
                                sequence = []
                                for j in range(0,seleframe*s,s):
                                    sequence.append(img_list[j+i]) #frames for a window size  
                                sequ_list.append(sequence)#adding sequence for sequence list
                                exp_list.append(exp)
                                expCount[exp+4] += seleframe#After sliding
                    else:
                        if step != 1:
                            m = img_len-seleframe
                            sequence = []
                            for j in range(seleframe):
                                sequence.append(img_list[j+m])#frames for a window size 
                            sequ_list.append(sequence)
                            exp_list.append(exp)
                            expCount[exp+4] += seleframe

    return sequ_list, exp_list, expCount


# In[11]:


def RawforVal(val):
    imgpath = "/*.bmp"
    sequ_list = []
    exp_list = []
    expCount = [0,0,0,0]      
    
    for folder, exp in zip(val[0],val[1]):      
        img_list = sorted(gb.glob(folder + imgpath))
        img_len = len(img_list)
        expCount[exp] += img_len
        sequ_list.append(img_list)
        exp_list.append(exp)
    return sequ_list, exp_list, expCount


# In[ ]:


"""
Created on Saturday April 13 10:51 2020
Used for data preprocessing without augmentation 
@author: Keira - github.com/Keira. Bai
"""
class MicroExpDataset(data.Dataset):
    """Micro Expressions dataset.""" 

    def __init__(self, folders, labels, tran_t = None, tran_r = None):
        
        self.tran_t = tran_t
        self.folders = folders
        self.labels = labels
        sequ_list = []
        exp_list = []
        exp = 0
        
    def __len__(self):
        return len(self.folders)         
    
    def __getitem__(self, idx):
        seq = self.folders[idx]
        exp = torch.LongTensor([self.labels[idx]])
        sample = []
        for i in seq:
            img = mpimg.imread(i)  
            img = Image.fromarray(img)
            img0 = self.tran_t(img) #original image
            sample.append(img0)
        sample = torch.stack(sample, dim=0) 
        return sample, exp

