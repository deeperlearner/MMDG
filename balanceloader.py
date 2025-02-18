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


def get_frame(path):
    
    
    frame =  Image.open(path)
    # frame = Image.fromarray(path)

    # face_frame = transform_face(frame)
    return frame



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
                 train = True , size = 224, cls = 'real'):
        
        # assert train_test_dev in ["Train", "Test", "Dev"]
        # assert ls in ["live", "spoof"]
        # assert protocol in [1,2,3,4]

        # self.root_dir = f"../OULU-NPU_all"
        # root = '/shared/domain-generalization-multi'
        self.allr = []
        self.alld = []
        self.alli = []
        self.atktype = [] # type_id 0 = real, others = attack
        self.cls = cls
        self.protocol = protocol

        for i in protocol:

            if i in ('train', 'train_grandtest', 'train_LOO_glasses',
                           'train_LOO_flexiblemask', 'train_LOO_rigidmask', 'train_LOO_prints',
                           'train_LOO_papermask', 'train_LOO_fakehead', 'train_LOO_replay'):
                
                print('train_get_data')
                r = sorted(glob.glob((os.path.join(root, f"train/"+ self.cls +"/rgb/*.jpg"))))
                d = sorted(glob.glob((os.path.join(root, f"train/"+ self.cls +"/depth/*.jpg"))))
                i = sorted(glob.glob((os.path.join(root, f"train/"+ self.cls +"/ir/*.jpg"))))
                
                self.allr += r
                self.alld += d
                self.alli += i
                print(len(self.allr), len(self.alld))

            if i == 'intraS':
                # r = sorted(glob.glob(os.path.join(root, f"train/"+ self.cls +"/color/*.jpg")))
                # d = sorted(glob.glob(os.path.join(root, f"train/"+ self.cls +"/depth/*.jpg")))
                # i = sorted(glob.glob(os.path.join(root, f"train/"+ self.cls +"/ir/*.jpg")))
                
                # self.allr += r
                # self.alld += d
                # self.alli += i
                
                self.allr = np.load(root + '/train/'+ self.cls +'/rgb.npy')
                self.alld = np.load(root + '/train/'+ self.cls +'/depth.npy')
                self.alli = np.load(root + '/train/'+ self.cls +'/ir.npy')
            
            if i == 'intraC':
                # r = sorted(glob.glob(os.path.join(root, f"train/"+ self.cls +"/profile/*.jpg")))
                # d = sorted(glob.glob(os.path.join(root, f"train/"+ self.cls +"/depth/*.jpg")))
                # i = sorted(glob.glob(os.path.join(root, f"train/"+ self.cls +"/ir/*.jpg")))
                # print(os.path.join(root, f"train/"+ self.cls +"/profile/*.jpg"))
                # self.allr += r
                # self.alld += d
                # self.alli += i
                self.allr = np.load(root + '/train/'+ self.cls +'/rgb.npy')
                self.alld = np.load(root + '/train/'+ self.cls +'/depth.npy')
                self.alli = np.load(root + '/train/'+ self.cls +'/ir.npy')
                
            if i == 'train':
                #/shared/flexible_multi_modality/
                # r = sorted(glob.glob((os.path.join(root, f"train/"+ self.cls +"/rgb/*.jpg"))))
                # d = sorted(glob.glob((os.path.join(root, f"train/"+ self.cls +"/depth/*.jpg"))))
                # i = sorted(glob.glob((os.path.join(root, f"train/"+ self.cls +"/ir/*.jpg"))))
                
                # self.allr += r
                # self.alld += d
                # self.alli += i
                
                self.allr = np.load(root + '/train/'+ self.cls +'/rgb.npy')
                self.alld = np.load(root + '/train/'+ self.cls +'/depth.npy')
                self.alli = np.load(root + '/train/'+ self.cls +'/ir.npy')
                
            if i == 'inter':
                
                self.allr = np.load(root + '/cross_testing/'+ self.cls +'/rgb.npy')
                self.alld = np.load(root + '/cross_testing/'+ self.cls +'/depth.npy')
                self.alli = np.load(root + '/cross_testing/'+ self.cls +'/ir.npy')
                
            if i == 'C':
                r = sorted(glob.glob(os.path.join(root, f"CeFA/"+ self.cls +"/profile/*.jpg")))
                d = sorted(glob.glob(os.path.join(root, f"CeFA/"+ self.cls +"/depth/*.jpg")))
                i = sorted(glob.glob(os.path.join(root, f"CeFA/"+ self.cls +"/ir/*.jpg")))
                
                #<client id> <session id> <presenter id> <type id> <pai id>.jpg
                
                self.allr += r
                self.alld += d
                self.alli += i
                print('train_get_data')
                # print(len(self.allr))
                
                # self.atktype += [int(0) for x in r]
            if i == 'S':
                r = sorted(glob.glob(os.path.join(root, f"SURF/"+ self.cls +"/profile/*.jpg")))
                d = sorted(glob.glob(os.path.join(root, f"SURF/"+ self.cls +"/depth/*.jpg")))
                i = sorted(glob.glob(os.path.join(root, f"SURF/"+ self.cls +"/ir/*.jpg")))
                

                
                #<client id> <session id> <presenter id> <type id> <pai id>.jpg
                
                self.allr += r
                self.alld += d
                self.alli += i
                print('train_get_data')
                # print(len(self.allr))
                
                # self.atktype += [ int(0) for x in r]
            if i == 'W':
                r = sorted(glob.glob(os.path.join(root, f"WMCA/"+ self.cls +"/profile/*.jpg")))
                d = sorted(glob.glob(os.path.join(root, f"WMCA/"+ self.cls +"/depth/*.jpg")))
                i = sorted(glob.glob(os.path.join(root, f"WMCA/"+ self.cls +"/ir/*.jpg")))

                #<client id> <session id> <presenter id> <type id> <pai id>.jpg
                
                self.allr += r
                self.alld += d
                self.alli += i
                print('train_get_data')
                # print(len(self.allr))
                
                # self.atktype += [int(x.split('/')[-1].split('.')[0].split('_')[3]) for x in r]
        

        if self.cls == 'real':
            # self.total_labels = np.zeros(len(self.allr), dtype=np.int64)
            self.total_labels = np.ones(len(self.allr), dtype=np.int64)
        else:
            # self.total_labels = np.ones(len(self.allr), dtype=np.int64)
            self.total_labels = np.zeros(len(self.allr), dtype=np.int64)


        self.atktype = np.array(self.total_labels)

        self.total_rgb = np.array(self.allr)
        self.total_depth = np.array(self.alld)
        self.total_ir = np.array(self.alli)
        
        # self.total_rgb = np.concatenate((self.all_liver, self.all_spoofr), axis=0)
        # self.total_depth = np.concatenate((self.all_lived, self.all_spoofd), axis=0)
        # self.total_ir = np.concatenate((self.all_livei, self.all_spoofi), axis=0)
        
        self.train = train
        self.size = size
        self.randomcrop = transforms.RandomResizedCrop(self.size)
        
    def transform(self, img1, img2, img3, train = True, size = 224):
        

        # Random crop
        # i, j, h, w = transforms.CenterCrop.get_params(
        #     img1, output_size=(224, 224))
        if train:
            img1 = TF.center_crop(TF.resize(img1, (256,256)), (size,size))
            img2 = TF.center_crop(TF.resize(img2, (256,256)), (size,size))
            img3 = TF.center_crop(TF.resize(img3, (256,256)), (size,size))
            # img1 = self.randomcrop(img1)
            # img2 = self.randomcrop(img2)
            # img3 = self.randomcrop(img3)

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
            # img1 = TF.resize(img1, (size,size))
            # img2 = TF.resize(img2, (size,size))
            # img3 = TF.resize(img3, (size,size))
            img1 = TF.center_crop(TF.resize(img1, (256,256)), (size,size))
            img2 = TF.center_crop(TF.resize(img2, (256,256)), (size,size))
            img3 = TF.center_crop(TF.resize(img3, (256,256)), (size,size))
            
            img2 = TF.rgb_to_grayscale(img2,num_output_channels=3)
            img3 = TF.rgb_to_grayscale(img3,num_output_channels=3)
            
        # Transform to tensor
        img1 = TF.to_tensor(img1)
        img2 = TF.to_tensor(img2)
        img3 = TF.to_tensor(img3)
        
        
        
        return img1, img2, img3

    def __getitem__(self, idx):
        
        # get rgb
        rgb = self.total_rgb[idx]
        # get depth
        depth = self.total_depth[idx]
        # get ir
        ir = self.total_ir[idx]
        
        labels = self.total_labels[idx]
        atktype = self.atktype[idx]
        
                
        # rgb_img = torch.stack(get_frame(rgb, self.transform_face)).transpose(0, 1)
        # depth_img = torch.stack(get_frame(depth, self.transform_face)).transpose(0, 1)
        # ir_img = torch.stack(get_frame(ir, self.transform_face)).transpose(0, 1)
        
        if self.protocol[0] == 'train' or self.protocol[0] == 'intraS':
            rgb = Image.fromarray(rgb)
            depth = Image.fromarray(depth)
            ir = Image.fromarray(ir)
        else:
            rgb = get_frame(rgb)
            depth = get_frame(depth)
            ir = get_frame(ir)
        
        rgb, depth, ir = self.transform(rgb, depth, ir, self.train, self.size)
        
                
        return rgb, depth, ir, labels, atktype


    def __len__(self):
        return len(self.total_rgb)


def get_loader( root, protocol, batch_size=10, shuffle=True, train = True, size = 224, cls = 'real'):
    
    _dataset = FAS_Dataset(root=root,
                           protocol=protocol,
                           train = train,
                           size = size,
                           cls = cls)
    
    return DataLoader(_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

def get_inf_iterator(data_loader):
    # """Inf data iterator."""
    while True:
        for rgb_img, depth_img, ir_img, labels, atktype in data_loader:
            yield (rgb_img, depth_img, ir_img, labels, atktype)

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
    
    train_loader = get_loader(root = '/var/mplab_share_data/domain-generalization-multi', protocol=['C'], batch_size=1800, shuffle=True)


    count = 0
    total = 0
    for i, (rgb_img, depth_img, ir_img, labels) in enumerate(train_loader):
        print(rgb_img.shape)
        print(depth_img.shape)
        print(ir_img.shape)
        total += rgb_img.shape[0]
        # print(type(rgb_img[0]))
        # cv2.imwrite('/home/Jxchong/Multi-Modality/rgb.jpg', rgb_img[0][9].permute(1,2,0).numpy()*255)
        # cv2.imwrite('/home/Jxchong/Multi-Modality/depth.jpg', depth_img[0][9].permute(1,2,0).numpy()*255)
        # cv2.imwrite('/home/Jxchong/Multi-Modality/ir.jpg', ir_img[0][9].permute(1,2,0).numpy()*255)
        count += torch.sum (labels)
        
    # print number of 1labels is tensor
    print('number of 1 labels: ' + str(count))
    
    print(total)