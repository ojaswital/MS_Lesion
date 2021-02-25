import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys
import os
from optparse import OptionParser
import numpy as np
from torch import optim
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
import config
import nibabel as nib
import itk
import SimpleITK as sitk

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# define dice coefficient
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

################################################ [TODO] ###################################################
# This function is used to evaluate the network after each epoch of training
# Input: network and validation dataset
# Output: average dice_coeff
def eval_net_batch(net, dataset, mask_list):
# set net mode to evaluation
    net.eval()
    with torch.no_grad():
        t1=time.time()
        reconstucted_mask = []
        original_mask = []
        mask_blank_list = []
        mapping = {}
        real_re_mask_list=[]
        real_re_mask_list2 = []
        real_mask_list=[]
        for i in range(len(mask_list)):
            mask = mask_list[i][0]
            mapping[mask_list[i][1]] = i
            mask_blank_list.append((np.zeros((mask.shape)), np.zeros((mask.shape))))
            original_mask.append(mask)
        # 这里需要考虑batch_size
        for i, b in enumerate(dataset):
            batch_img = b['img'].to(device)#b*c*64*64*64
            batch_index = b['index']#b
            batch_d = b['d']
            batch_w = b['w']
            batch_h = b['h']
            # Feed the image to network to get predicted mask
            batch_mask_pred = net(batch_img)#b*1*64*64*64
            for ii in range(batch_mask_pred.shape[0]):
                index = batch_index[ii]
                mask_pred = batch_mask_pred[ii]
                d = batch_d[ii]
                w = batch_w[ii]
                h = batch_h[ii]
                mask_blank_list[mapping[index.item()]][0][0:1, d:d + 64, w:w + 64, h:h + 64] += mask_pred.detach().squeeze(0).detach().cpu().numpy()
                mask_blank_list[mapping[index.item()]][1][0:1, d:d + 64, w:w + 64, h:h + 64] += 1
        for i in range(len(mask_blank_list)):
            mask_blank_list[i][1][mask_blank_list[i][1] == 0] = 1
            reconstucted_mask.append(mask_blank_list[i][0] / mask_blank_list[i][1])
        #majority vote or just add for every 3 reconstructed_mask
        for i in range(0,len(reconstucted_mask),3):
            temp0=reconstucted_mask[i + 0]
            temp1=reconstucted_mask[i + 1].transpose(0, 3, 1, 2)#2,0,1
            temp2=reconstucted_mask[i + 2].transpose(0, 2, 3, 1)#1,2,0
            #if add
            # real_re_mask=(temp0+temp1+temp2)/3
            # real_re_mask_list.append(real_re_mask)
            # real_mask_list.append(mask_list[i][0])
            #if vote
            tt=np.zeros(temp0.shape)
            for d in range(tt.shape[1]):
                for w in range(tt.shape[2]):
                    for h in range(tt.shape[3]):
                        if (temp0[0][d][w][h]>0.5 and temp1[0][d][w][h]>0.5) or (temp0[0][d][w][h]>0.5 and temp2[0][d][w][h]>0.5)\
                            or (temp1[0][d][w][h]>0.5 and temp2[0][d][w][h]>0.5):
                            tt[0][d][w][h]=1
            real_re_mask_list2.append(tt)
    # return dice_coeff_cpu(reconstucted_mask, original_mask)
    return dice_coeff_cpu(real_re_mask_list2, real_mask_list)

if __name__=="__main__":
    # get all the image and mask path and number of images

    patch_img_mask_list = []
    train_img_masks=[]
    val_img_masks=[]
    train_index = []
    val_index = []
    mask_list = []
    voxel = 1.0 #64 * 64 * 64 * 0.0001
    index = 0
    ori_train_img_list = []
    ori_train_mask_list = []
    ori_test_img_list = []
    ori_test_mask_list = []
    new_h, new_w, new_d = 64, 64, 64  # 80,100 288,384 96,72
    #voxel=64*64*64*0.0001

    for root, dirs, files in os.walk("ISBI_Dataset/"):
        if dirs == []:
            img = np.array(nib.load(os.path.join(root, "FLAIR_preprocessed.nii.gz")).get_fdata())
            max_img = np.max(img)
            img = img / max_img
            ori_test_img_list.append(img)
            mask = np.array(nib.load(os.path.join(root, "Consensus.nii.gz")).get_fdata())
            ori_test_mask_list.append(mask)
            
            
    for img, mask in zip(ori_train_img_list, ori_train_mask_list):
        # pad zero
        pad_size = [0, 0, 0]
        div = [64, 64, 64]
        pad = False
        for i in range(len(img.shape)):
            remain = img.shape[i] % div[i]
            if remain != 0:
                pad = True
                pad_size[i] = (img.shape[i] // div[i] + 1) * div[i] - img.shape[i]
        if pad:
            # deal with odd number of padding
            pad0 = (pad_size[0] // 2, pad_size[0] - pad_size[0] // 2)
            pad1 = (pad_size[1] // 2, pad_size[1] - pad_size[1] // 2)
            pad2 = (pad_size[2] // 2, pad_size[2] - pad_size[2] // 2)
            img = np.pad(img, (pad0, pad1, pad2), 'constant')
            mask = np.pad(mask, (pad0, pad1, pad2), 'constant')
        depth = mask.shape[0]
        width = mask.shape[1]
        height = mask.shape[2]
        for d in range(0, depth - 64 + 16, 16):
            for w in range(0, width - 64 + 16, 16):
                for h in range(0, height - 64 + 16, 16):
                    patch_img = img[d:d + 64, w:w + 64, h:h + 64]
                    patch_mask = mask[d:d + 64, w:w + 64, h:h + 64]
                    if np.sum(patch_mask) >=voxel:
                        patch_img = np.expand_dims(patch_img, 0)
                        patch_mask = np.expand_dims(patch_mask, 0)
                        train_img_masks.append((patch_img, patch_mask, index, d, w, h))
        mask = np.expand_dims(mask, 0)
        mask_list.append((mask, index))
        train_index.append(index)
        index += 1

    for img, mask in zip(ori_test_img_list, ori_test_mask_list):
        # pad zero
        pad_size = [0, 0, 0]
        div = [64, 64, 64]
        pad = False
        for i in range(len(img.shape)):
            remain = img.shape[i] % div[i]
            if remain != 0:
                pad = True
                pad_size[i] = (img.shape[i] // div[i] + 1) * div[i] - img.shape[i]
        if pad:
            # deal with odd number of padding
            pad0 = (pad_size[0] // 2, pad_size[0] - pad_size[0] // 2)
            pad1 = (pad_size[1] // 2, pad_size[1] - pad_size[1] // 2)
            pad2 = (pad_size[2] // 2, pad_size[2] - pad_size[2] // 2)
            img = np.pad(img, (pad0, pad1, pad2), 'constant')
            mask = np.pad(mask, (pad0, pad1, pad2), 'constant')
        depth = mask.shape[0]
        width = mask.shape[1]
        height = mask.shape[2]
        for d in range(0, depth - 64 + 16, 16):
            for w in range(0, width - 64 + 16, 16):
                for h in range(0, height - 64 + 16, 16):
                    patch_img = img[d:d + 64, w:w + 64, h:h + 64]
                    patch_mask = mask[d:d + 64, w:w + 64, h:h + 64]
                    patch_img = np.expand_dims(patch_img, 0)
                    patch_mask = np.expand_dims(patch_mask, 0)
                    val_img_masks.append((patch_img, patch_mask, index, d, w, h))
        mask = np.expand_dims(mask, 0)
        mask_list.append((mask, index))
        val_index.append(index)
        index += 1

    real_train_mask_list = []
    real_test_mask_list = []
    for mask, index in mask_list:
        if index in train_index:
            real_train_mask_list.append((mask, index))
        if index in val_index:
            real_test_mask_list.append((mask, index))
    val_dataset = CustomDataset(val_img_masks, transforms=transforms.Compose([ToTensor()]))

    ################################################ [TODO] ###################################################
    # Create a UNET object from the class defined above. Input channels = 3, output channels = 1
    #net= UNet3D(1, 1)
    net=get_pose_net(config.cfg, 0)
    net.to(device)
    # run net.to(device) if using GPU
    # If continuing from previously saved model, use
    net.load_state_dict(torch.load('./results_HRNet_ISBI/3_flod/model_best.pth',map_location='cuda:0'),False)
    #print(net)

    # This shows the number of parameters in the network
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Number of parameters in network: ', n_params)



    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0)
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    
    #val_dice = eval_net(net, train_loader, real_train_img_mask_list)
    #print('Validation Dice Coeff: {}'.format(val_dice))

    val_dice = eval_net_batch(net, val_loader, real_test_mask_list)
    print('Validation Dice Coeff: {}'.format(val_dice))
