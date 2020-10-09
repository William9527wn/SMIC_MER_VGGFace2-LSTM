#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Data preprocessing version 4.0
#1. Random steps between frames within a slide window;
#2. Variable sele_frame(slide window size)
#3. Fixed-size of slide window(if image length smaller than the seleframe,adding frames);
#4. Mirror flipping
#5. Raw data for validation


# In[2]:


"""
Created on Monday May 18 10:51 2020
Used to pre-process dataset 
@author: Keira - github.com/Keira. Bai
a.function to divide data set into 5 groups for cross-validation, return a 5*Ndimention matrix 
b.function to extract trainiing data into fixed size sequences, sequences consist by consecuous internal
c.function to extract validation data, sequences consist by raw image and adding frames to the same size
d.function to read image
"""
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


# In[3]:


#Allocate dataset into k groups for training and validation
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
    #Load in all folders adding expression
    for vid_dir in [neg_dir, pos_dir, sur_dir, non_dir]:        
        sequ = gb.glob(datapaths+vid_dir)#read all folders under different expression
        sequ_list += sequ 
        label_list += [label for i in range(len(sequ))] #read in the same number of expression label
        label += 1 #change the expression category
    #Devide All Data into k groups
    for i in range(k):
        if k-i>1:
            train_list, valid_list, train_label, valid_label = train_test_split(sequ_list, label_list,                                           test_size=1/(k-i), random_state=42)
            Val_tra_tes_list.append([valid_list, valid_label])
            
        else:
            Val_tra_tes_list.append([train_list, train_label])

        sequ_list = train_list
        label_list = train_label   

    return Val_tra_tes_list 


# In[4]:


#Data augmentation on sequence level by fixed size
def InputImagewithSlide(tra_tes_list, seleframe = 11):    
    imgpath = "/*.bmp"
    sequ_list = []
    
    exp_list = []
    step_list = [1,2,3,4,5,6]
    expCount = [0,0,0,0,0,0,0,0]

    for group in tra_tes_list:    
        for folder, exp in zip(group[0], group[1]):
            img_list = sorted(gb.glob(folder + imgpath))#get frames from the same one folder
            img_len = len(img_list)
            expCount[exp] += img_len#recording the same number of expression 
            padSeq = []
            #repackage training sequence            
            if img_len < seleframe: #for frames less than seleframe
                padSeq.extend(img_list)
                padSeq.extend(img_list[:seleframe-img_len])
                sequ_list.append(padSeq)#append the sequence directly
                exp_list.append(exp)
                expCount[exp+4] += seleframe
            else: #for frames more than seleframe
                #control step to keep the amout balance between expressions
                if exp == 3:#non-micro-expression
                    step = 3#decrease the growth rate
                else:
                    step = 1
                for i in range(0, img_len-seleframe+step, step):
                        if ((img_len-i) >= (seleframe+step-1)):#sequence is enough for a seleframe+step
                            for s in step_list:#assign inner step for frame
                                if img_len-i >= seleframe*s: #if frame number is enough for the inner step
                                    sequence = []
                                    for j in range(0,seleframe*s,s):
                                        sequence.append(img_list[j+i]) #frames for a window size  
                                    sequ_list.append(sequence)#adding sequence for sequence list
                                    exp_list.append(exp)
                                    expCount[exp+4] += seleframe
                        else:
                            if step != 1:
                                m = img_len-seleframe
                                sequence = []
                                for j in range(seleframe):
                                    sequence.append(img_list[j+m])#adding the frames in the end part of video 
                                sequ_list.append(sequence)
                                exp_list.append(exp)
                                expCount[exp+4] += seleframe

    return sequ_list, exp_list, expCount


# In[5]:


#Sorting out frames for valifation
def InputImageforVal(val,seleframe):
    imgpath = "/*.bmp"
    sequ_list = []
    exp_list = []
    expCount = [0,0,0,0]  
    starPoint = 0
    endPoint = 0

    for folder, exp in zip(val[0],val[1]): #val[0] for images, val[1] for expressions
        img_list = sorted(gb.glob(folder + imgpath))#reading all images in one folder
        img_len = len(img_list)
        expCount[exp] += img_len #recording the number of each expression
        #repackage images in each folder in fixed-length sequences
        for k in range(0,img_len, seleframe):#seleframe is also the increase step
            sequence = img_list[k:k+seleframe]
            if len(sequence) < seleframe:                
                sequence = img_list[img_len-seleframe:img_len]
            sequ_list.append(sequence)
            exp_list.append(exp)
    
    return sequ_list, exp_list, expCount


# In[6]:


#reloader the dataset after flipping 
def Reloader(train_set):
    train_dataset = []
    for Tset in train_set:
        Ro_len = len(Tset[0])
        for i in range(Ro_len):
            train_dataset.append([Tset[0][i],Tset[1][i]])
    return train_dataset


# In[7]:


#Raw data for validation
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


# In[8]:


#implementing image data to matrix
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
        sampleo = []
        samplef = []
        for i in seq:
            img = mpimg.imread(i)#read image
            img = Image.fromarray(img)#transfer array image to matrix
            #get mirror flip
            imgf = transforms.functional.hflip(img)
            imgf = self.tran_t(imgf)
            samplef.append(imgf)
            #original image
            imgo = self.tran_t(img)             
            sampleo.append(imgo)            

        sampleo = torch.stack(sampleo, dim=0) 
        samplef = torch.stack(samplef, dim=0) 

        return sampleo, samplef, exp


# In[9]:


# datapaths = "../SMIC/SMIC_all_cropped/HS/*/" 
# k=5
# v=0
# seleframe = 22
# Val_tra_tes_list = CrossAllocation(datapaths,k)
# batch_size = 40
# # Detect devices
# use_cuda = torch.cuda.is_available()                   # check if GPU exists
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # use CPU or GPU

# params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}

# #Set transformation
# tran_t = transforms.Compose(
#     [transforms.Resize([224,224]), 
#      transforms.ToTensor(),     
#     ])
# val = Val_tra_tes_list[v] 
# valid_list, valid_label, val_expCount = InputImageforVal(val, seleframe)
# # for v in valid_list:
# #     print(len(v))
# tra_tes_list = np.delete(Val_tra_tes_list,v,axis = 0) 
# train_test_list, train_test_label, expCount = InputImagewithSlide(tra_tes_list, seleframe)#get training and testing data
# # train_list, test_list, train_label, test_label = train_test_split(train_test_list, train_test_label, \
# #                                       test_size=1/k, random_state=42)
# # train_set, test_set, valid_set = MicroExpDataset(train_list[:3], train_label[:3], tran_t=tran_t), \
# #                                 MicroExpDataset(test_list, test_label, tran_t=tran_t), \
# #                                 MicroExpDataset(valid_list, valid_label, tran_t=tran_t)
# # train_loader = data.DataLoader(train_set, **params)
# # # test_loader = data.DataLoader(test_set, **params)
# # # valid_loader = data.DataLoader(valid_set, **params)

