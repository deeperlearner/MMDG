from __future__ import print_function, division
import os
import torch

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import glob
import random
import numpy as np
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
import cv2


def get_frame(path):
    
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    return img



def getSubjects(configPath):
    
    f = open(configPath, "r")
    
    all_live, all_spoof = [], []
    while(True):
        line = f.readline()
        if not line:
            break
        line = line.strip()
        # print(line)
        
        ls, subj = line.split(",")
        if(ls == "+1"):
            all_live.append(subj)
            # print("live", subj)
        else:
            all_spoof.append(subj)
            # print("spoof", subj)
    
    print(f"{configPath=}")
    print(f"{len(all_live)=}, {len(all_spoof)=}")
    
    return all_live, all_spoof



class FAS_Dataset(Dataset):
    def __init__(self, root, 
                 protocol=['C','S','W','train', 'train_grandtest', 'train_LOO_glasses',
                           'train_LOO_flexiblemask', 'train_LOO_rigidmask', 'train_LOO_prints',
                           'train_LOO_papermask', 'train_LOO_fakehead', 'train_LOO_replay'], 
                 mode='train', size=256, cls_type='live'):
        
        # assert cls_type in ["live", "spoof"]
        self.allr = []
        self.alld = []
        self.alli = []
        self.labels = []
        # self.atktype = [] # type_id 0 = real, others = attack
        self.mode = mode
        self.size = size
        self.cls = cls_type
        self.protocol = protocol

        for i in protocol:
            if i == 'train_list.txt':
                data_path = os.path.join(root, self.mode + '.txt')
                with open(data_path, 'r') as f:
                    for line in f.readlines():
                        rgb, depth, ir, label = line.strip().split(' ')
                        label = int(label)
                        if self.mode == 'train':
                            if self.cls == 'live':
                                collect = True if label == 1 else False
                            elif self.cls == 'spoof':
                                collect = True if label == 0 else False
                            if collect:
                                rgb = os.path.join(root, rgb)
                                depth = os.path.join(root, depth)
                                ir = os.path.join(root, ir)
                                self.allr.append(rgb)
                                self.alld.append(depth)
                                self.alli.append(ir)
                                self.labels.append(label)
                        else:
                            rgb = os.path.join(root, rgb)
                            depth = os.path.join(root, depth)
                            ir = os.path.join(root, ir)
                            self.allr.append(rgb)
                            self.alld.append(depth)
                            self.alli.append(ir)
                            self.labels.append(label)
                self.total_rgb = self.allr
                self.total_depth = self.alld
                self.total_ir = self.alli
                self.total_labels = self.labels
            elif i == 'val_list.txt':
                data_path = os.path.join(root, i)
                with open(data_path, 'r') as f:
                    for line in f.readlines():
                        rgb, depth, ir, label = line.strip().split(' ')
                        label = int(label)
                        rgb = os.path.join(root, rgb)
                        depth = os.path.join(root, depth)
                        ir = os.path.join(root, ir)
                        self.allr.append(rgb)
                        self.alld.append(depth)
                        self.alli.append(ir)
                        self.labels.append(label)
                self.total_rgb = self.allr
                self.total_depth = self.alld
                self.total_ir = self.alli
                self.total_labels = self.labels

    def transform(self, img1, img2, img3):
        # Random crop
        # i, j, h, w = transforms.CenterCrop.get_params(
        #     img1, output_size=(224, 224))
        # randomcrop = transforms.RandomResizedCrop(self.size)
        if self.mode == 'train':
            img1 = TF.center_crop(TF.resize(img1, (256,256)), (self.size, self.size))
            img2 = TF.center_crop(TF.resize(img2, (256,256)), (self.size, self.size))
            img3 = TF.center_crop(TF.resize(img3, (256,256)), (self.size, self.size))
            # img1 = randomcrop(img1)
            # img2 = randomcrop(img2)
            # img3 = randomcrop(img3)

            img2 = TF.rgb_to_grayscale(img2,num_output_channels=3)
            img3 = TF.rgb_to_grayscale(img3,num_output_channels=3)

            if random.random() > 0.5:
                img1 = TF.hflip(img1)
                img2 = TF.hflip(img2)
                img3 = TF.hflip(img3)

            # Random vertical flipping
            if random.random() > 0.5:
                img1 = TF.vflip(img1)
                img2 = TF.vflip(img2)
                img3 = TF.vflip(img3)

            # Random rotation
            angle = transforms.RandomRotation.get_params(degrees=(-30, 30))
            img1 = TF.rotate(img1,angle)
            img2 = TF.rotate(img2,angle)
            img3 = TF.rotate(img3,angle)
        else:
            # img1 = TF.resize(img1, (self.size, self.size))
            # img2 = TF.resize(img2, (self.size, self.size))
            # img3 = TF.resize(img3, (self.size, self.size))
            img1 = TF.center_crop(TF.resize(img1, (256,256)), (self.size, self.size))
            img2 = TF.center_crop(TF.resize(img2, (256,256)), (self.size, self.size))
            img3 = TF.center_crop(TF.resize(img3, (256,256)), (self.size, self.size))
            
            img2 = TF.rgb_to_grayscale(img2,num_output_channels=3)
            img3 = TF.rgb_to_grayscale(img3,num_output_channels=3)
            
        # Transform to tensor
        img1 = TF.to_tensor(img1)
        img2 = TF.to_tensor(img2)
        img3 = TF.to_tensor(img3)
        
        return img1, img2, img3

    def __getitem__(self, idx):
        
        # get rgb
        rgb_path = self.total_rgb[idx]
        # get depth
        depth_path = self.total_depth[idx]
        # get ir
        ir_path = self.total_ir[idx]
        
        labels = self.total_labels[idx]
        # atktype = self.atktype[idx]
        
        rgb = get_frame(rgb_path)
        depth = get_frame(depth_path)
        ir = get_frame(ir_path)
        
        rgb, depth, ir = self.transform(rgb, depth, ir)

        return rgb, depth, ir, labels

    def __len__(self):
        return len(self.total_rgb)


def get_loader(root, protocol, batch_size=10, shuffle=True, mode='train', size=256, cls='live'):
    
    _dataset = FAS_Dataset(root, protocol=protocol, mode=mode, size=size, cls_type=cls)
    
    return DataLoader(_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

def get_inf_iterator(data_loader):
    # """Inf data iterator."""
    while True:
        for rgb_img, depth_img, ir_img, labels in data_loader:
            yield (rgb_img, depth_img, ir_img, labels)

# def collate_batch(batch):
       
#     face_frames_list, image_paths_list, bg_frames_list, bg_paths_list = [], [], [], []
    
#     for (face_frames, image_paths, bg_frames, bg_paths) in batch:
        
#         face_frames_list.append(face_frames)
#         image_paths_list.append(image_paths)
#         bg_frames_list.append(bg_frames)
#         bg_paths_list.append(bg_paths)

#     return face_frames_list, image_paths_list, bg_frames_list, bg_paths_list



import cv2
if __name__ == "__main__":
    
    live_loader = get_loader(root='/media/back/home/chuck/Dataset/CASIA_SURF/challenge/train', protocol=['train_list.txt'], batch_size=1800, shuffle=True, mode='train', size=256, cls='live')
    spoof_loader = get_loader(root='/media/back/home/chuck/Dataset/CASIA_SURF/challenge/train', protocol=['train_list.txt'], batch_size=1800, shuffle=True, mode='train', size=256, cls='spoof')
    live_loader = get_loader(root='/media/back/home/chuck/Dataset/CASIA_SURF/challenge/train', protocol=['train_list.txt'], batch_size=1800, shuffle=True, mode='valid', size=256, cls='live')
    spoof_loader = get_loader(root='/media/back/home/chuck/Dataset/CASIA_SURF/challenge/train', protocol=['train_list.txt'], batch_size=1800, shuffle=True, mode='valid', size=256, cls='spoof')
    live_loader = get_loader(root='/media/back/home/chuck/Dataset/CASIA_SURF/challenge/train', protocol=['train_list.txt'], batch_size=1800, shuffle=True, mode='test', size=256, cls='live')
    spoof_loader = get_loader(root='/media/back/home/chuck/Dataset/CASIA_SURF/challenge/train', protocol=['train_list.txt'], batch_size=1800, shuffle=True, mode='test', size=256, cls='spoof')
    os._exit(0)

    count = 0
    total = 0
    for i, (rgb_img, depth_img, ir_img, labels) in enumerate(live_loader):
        print(rgb_img.shape)
        print(depth_img.shape)
        print(ir_img.shape)
        total += rgb_img.shape[0]
        # print(type(rgb_img[0]))
        # cv2.imwrite('/home/Jxchong/Multi-Modality/rgb.jpg', rgb_img[0][9].permute(1,2,0).numpy()*255)
        # cv2.imwrite('/home/Jxchong/Multi-Modality/depth.jpg', depth_img[0][9].permute(1,2,0).numpy()*255)
        # cv2.imwrite('/home/Jxchong/Multi-Modality/ir.jpg', ir_img[0][9].permute(1,2,0).numpy()*255)
        count += torch.sum(labels)
        
    # print number of 1labels is tensor
    print('number of 1 labels: ' + str(count))
    
    print(total)
