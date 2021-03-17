import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys
import os
from optparse import OptionParser
import numpy as np
from torch import optim
# Test the ISBI dataset on the model
from PIL import Image
from torch.autograd import Function, Variable
import matplotlib.pyplot as plt
import matplotlib
from torchvision import transforms
import glob
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset
import cv2
from utils.util import *
from Model.HRNet_3D import  *
from Model.UNet_3D import *
import config
import nibabel as nib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def dice_coeff_cpu(prediction, target):
    s=[]
    eps = 1.0
    for i, (a, b) in enumerate(zip(prediction, target)):
        A = a.flatten()
        B = b.flatten()
        inter = np.dot(A, B)
        union = np.sum(A) + np.sum(B) + eps
        # Calculate DICE
        d = (2 * inter+eps) / union
        s.append(d)
    return s

if __name__=="__main__":
    train_img_masks, val_img_masks, real_train_mask_list, real_test_mask_list = load_data_aug('../ISBI2015/', 2)
    print("successfully loaded data")
    train_dataset = CustomDataset(train_img_masks, transforms=transforms.Compose([ToTensor()]))
    val_dataset = CustomDataset(val_img_masks, transforms=transforms.Compose([ToTensor()]))
    #net=get_pose_net(config.cfg, 1)
    net = torch.nn.DataParallel(get_pose_net(config.cfg, 1,1), device_ids=[0, 1])
    # net=UNet3D(1,1)
    net.to(device)

    # net.load_state_dict(torch.load('results_HRNet_ISBI_march/1_flod/model_best.pth',map_location='cuda:1'))
    # net.load_state_dict(torch.load('results_HRNet_ISBI_march/1_flod/model_best.pth'))
    load_checkpoint('results_HRNet_ISBI_march/3_flod/model_best.pth',net,None)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    val_dice = eval_net(net, train_loader, real_train_mask_list,device)
    print('Validation Dice Coeff: {}'.format(val_dice))

    val_dice = eval_net(net, val_loader, real_test_mask_list,device)
    print('Validation Dice Coeff: {}'.format(val_dice))
