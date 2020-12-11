# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""Entry point for testing AttGAN network with manipulating multiple attributes."""
''' A demo for manipulating face attributes on images in CelebA database'''
''' Only support some of the attributes in list_attr_celeba.txt; you could add other attributes if you want. '''
''' Only choose a few images from CelebA database; you could add other images if you want. '''

import argparse
import json
import os
from os.path import join

import torch
import torch.utils.data as data
import torchvision.utils as vutils

from attgan import AttGAN
from helpers import Progressbar
from utils import find_model
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


def nothing(x):
    pass

def UnNorm(tensor, mean, std):
    tens = tensor.mul(std).add(mean)
    # print(tensor)
    tens = tens.mul_(255)
        # The normalize code -> t.sub_(m).div_(s)
    # print(tensor)
    return tens

def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='256_shortcut1_inject1_none')
    parser.add_argument('--data_path', type = str, default = '../dataset/img_align_celeba')
    parser.add_argument('--attr_path', type = str, default = '../dataset/list_attr_celeba.txt')
    parser.add_argument('--load_epoch', dest='load_epoch', type=str, default='latest')
    parser.add_argument('--gpu', action='store_true')
    return parser.parse_args(args)

args_ = parse()

with open(join('output', args_.experiment_name, 'setting.txt'), 'r') as f:
    args = json.load(f, object_hook=lambda d: argparse.Namespace(**d))


args.load_epoch = args_.load_epoch
args.gpu = args_.gpu

attgan = AttGAN(args)
attgan.load(find_model(join('output', args.experiment_name, 'checkpoint'), args.load_epoch))
attgan.eval()

progressbar = Progressbar()

tf = transforms.Compose([
    transforms.CenterCrop(170),
    transforms.Resize(args.img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

att_list = open(args_.attr_path, 'r', encoding='utf-8').readlines()[0].split()
# print(att_list)
atts = [att_list.index(att) + 1 for att in args.attrs]
images = np.loadtxt(args_.attr_path, skiprows=2, usecols=[0], dtype=np.str)
labels = np.loadtxt(args_.attr_path, skiprows=2, usecols=atts, dtype=np.int)

while True:

    try:
        num_in=int(input('Please input a number from 1-12?'))
    except ValueError:
        print ("It is not a number!")
        continue

    # 1827 1833  1838 1848 1852 1854 1856 1859 1861 1865 1868 1883

    if num_in == 1: number = 1827
    elif num_in == 2: number = 1833
    elif num_in == 3: number = 1838
    elif num_in == 4: number = 1848
    elif num_in == 5: number = 1852
    elif num_in == 6: number = 1854
    elif num_in == 7: number = 1856
    elif num_in == 8: number = 1859
    elif num_in == 9: number = 1861
    elif num_in == 10: number = 1865
    elif num_in == 11: number = 1868
    elif num_in == 12: number = 1883
    else: 
        print("The number is not within the range!")
        continue

    image = images[number:number+1]  # 182639
    label = labels[number:number+1]
    index = 0
    # pdb.set_trace()
    img_a = tf(Image.open(os.path.join(args_.data_path, image[index])))
    att_a = torch.tensor((label[index] + 1) // 2)
    att_a = att_a.unsqueeze(0)
    img_a = img_a.unsqueeze(0)

    cv2.namedWindow('Face Attribute Manipulation')

    cv2.createTrackbar('Bald', 'Face Attribute Manipulation', att_a[0][0]*10, 10, nothing)
    cv2.createTrackbar('Bangs', 'Face Attribute Manipulation', att_a[0][1]*10, 10, nothing)
    cv2.createTrackbar('Black_Hair', 'Face Attribute Manipulation', att_a[0][2]*10, 10, nothing)
    cv2.createTrackbar('Blond_Hair', 'Face Attribute Manipulation', att_a[0][3]*10, 10, nothing)
    cv2.createTrackbar('Brown_Hair', 'Face Attribute Manipulation', att_a[0][4]*10, 10, nothing)
    cv2.createTrackbar('Bushy_Eyebrows', 'Face Attribute Manipulation', att_a[0][5]*10, 10, nothing)
    cv2.createTrackbar('Eyeglasses', 'Face Attribute Manipulation', att_a[0][6]*10, 10, nothing)
    cv2.createTrackbar('Male', 'Face Attribute Manipulation', att_a[0][7]*10, 10, nothing)
    cv2.createTrackbar('Mouth_Slightly_Open', 'Face Attribute Manipulation', att_a[0][8]*10, 10, nothing)
    cv2.createTrackbar('Mustache', 'Face Attribute Manipulation', att_a[0][9]*10, 10, nothing)
    cv2.createTrackbar('No_Beard', 'Face Attribute Manipulation', att_a[0][10]*10, 10, nothing)
    cv2.createTrackbar('Pale_Skin', 'Face Attribute Manipulation', att_a[0][11]*10, 10, nothing)
    cv2.createTrackbar('Young', 'Face Attribute Manipulation', att_a[0][12]*10, 10, nothing)


    img_a = img_a.cuda() if args.gpu else img_a
    att_a = att_a.cuda() if args.gpu else att_a
    att_a = att_a.type(torch.float)
    att_b = att_a.clone()
    img_ori = img_a


    while(True):
       
        cv2.imshow('Face Attribute Manipulation', UnNorm(img_a, 0.5, 0.5).permute(0,2,3,1).squeeze(0).cpu().numpy().astype(np.uint8)[:,:,[2,1,0]])    

        try:
            if cv2.waitKey(50) == 27:
                cv2.destroyAllWindows()
                break
        except:
            pass

        att_b[0][0] = cv2.getTrackbarPos('Bald', 'Face Attribute Manipulation') / 10
        att_b[0][1] = cv2.getTrackbarPos('Bangs', 'Face Attribute Manipulation') / 10
        att_b[0][2] = cv2.getTrackbarPos('Black_Hair', 'Face Attribute Manipulation') / 10
        att_b[0][3] = cv2.getTrackbarPos('Blond_Hair', 'Face Attribute Manipulation') / 10
        att_b[0][4] = cv2.getTrackbarPos('Brown_Hair', 'Face Attribute Manipulation') / 10
        att_b[0][5] = cv2.getTrackbarPos('Bushy_Eyebrows', 'Face Attribute Manipulation') / 10
        att_b[0][6] = cv2.getTrackbarPos('Eyeglasses', 'Face Attribute Manipulation') / 10
        att_b[0][7] = cv2.getTrackbarPos('Male', 'Face Attribute Manipulation') / 10
        att_b[0][8] = cv2.getTrackbarPos('Mouth_Slightly_Open', 'Face Attribute Manipulation') / 10
        att_b[0][9] = cv2.getTrackbarPos('Mustache', 'Face Attribute Manipulation') / 10
        att_b[0][10] = cv2.getTrackbarPos('No_Beard', 'Face Attribute Manipulation') / 10
        att_b[0][11] = cv2.getTrackbarPos('Pale_Skin', 'Face Attribute Manipulation') / 10
        att_b[0][12] = cv2.getTrackbarPos('Young', 'Face Attribute Manipulation') / 10
        
        with torch.no_grad():
            att_b_ = (att_b * 2 - 1) * args.thres_int
            img_a = attgan.G(img_ori, att_b_)
            _, att_test = attgan.D(img_a)

            