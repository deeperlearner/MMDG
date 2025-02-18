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
    
    # if path is .npy file then
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
                protocol=['C','S','W','train', 'grandtest', 'LOO_glasses',
                           'LOO_flexiblemask', 'LOO_rigidmask', 'LOO_prints',
                           'LOO_papermask', 'LOO_fakehead', 'LOO_replay'], 
                train = True , size = 224):
        
        # assert train_test_dev in ["Train", "Test", "Dev"]
        # assert ls in ["live", "spoof"]
        # assert protocol in [1,2,3,4]

        # self.root_dir = f"../OULU-NPU_all"
        # root = '/shared/domain-generalization-multi'

        self.all_liver = []
        self.all_lived = []
        self.all_livei = []
        self.all_spoofr = []
        self.all_spoofd = []
        self.all_spoofi = []
        self.protocol = protocol

        for i in protocol:

            if i in ('train', 'grandtest', 'LOO_glasses',
                           'LOO_flexiblemask', 'LOO_rigidmask', 'LOO_prints',
                           'LOO_papermask', 'LOO_fakehead', 'LOO_replay'):
                print('get_test_data')
                
                liver = sorted(glob.glob(os.path.join(root, f"test/real/rgb/*.jpg")))
                lived = sorted(glob.glob(os.path.join(root, f"test/real/depth/*.jpg")))
                livei = sorted(glob.glob(os.path.join(root, f"test/real/ir/*.jpg")))
                
                spoofr = sorted(glob.glob(os.path.join(root, f"test/spoof/rgb/*.jpg")))
                spoofd = sorted(glob.glob(os.path.join(root, f"test/spoof/depth/*.jpg")))
                spoofi = sorted(glob.glob(os.path.join(root, f"test/spoof/ir/*.jpg")))
                
                self.all_liver += liver
                self.all_lived += lived
                self.all_livei += livei
                
                self.all_spoofr += spoofr
                self.all_spoofd += spoofd
                self.all_spoofi += spoofi
                
            if i == 'inter':
                # liver = sorted(glob.glob(os.path.join(root, f"cross_testing/real/rgb/*.jpg")))
                # lived = sorted(glob.glob(os.path.join(root, f"cross_testing/real/depth/*.jpg")))
                # livei = sorted(glob.glob(os.path.join(root, f"cross_testing/real/ir/*.jpg")))
                
                # spoofr = sorted(glob.glob(os.path.join(root, f"cross_testing/spoof/rgb/*.jpg")))
                # spoofd = sorted(glob.glob(os.path.join(root, f"cross_testing/spoof/depth/*.jpg")))
                # spoofi = sorted(glob.glob(os.path.join(root, f"cross_testing/spoof/ir/*.jpg")))
                
                # self.all_liver += liver
                # self.all_lived += lived
                # self.all_livei += livei
                
                # self.all_spoofr += spoofr
                # self.all_spoofd += spoofd
                # self.all_spoofi += spoofi
                
                self.all_liver = np.load(root + '/cross_testing/real/rgb.npy')
                self.all_lived = np.load(root + '/cross_testing/real/depth.npy')
                self.all_livei = np.load(root + '/cross_testing/real/ir.npy')
                self.all_spoofr= np.load(root + '/cross_testing/spoof/rgb.npy')
                self.all_spoofd= np.load(root + '/cross_testing/spoof/depth.npy')
                self.all_spoofi= np.load(root + '/cross_testing/spoof/ir.npy')
                
            if i == 'C':
                liver = sorted(glob.glob(os.path.join(root, f"CeFA/real/profile/*.jpg")))
                lived = sorted(glob.glob(os.path.join(root, f"CeFA/real/depth/*.jpg")))
                livei = sorted(glob.glob(os.path.join(root, f"CeFA/real/ir/*.jpg")))
                
                spoofr = sorted(glob.glob(os.path.join(root, f"CeFA/spoof/profile/*.jpg")))
                spoofd = sorted(glob.glob(os.path.join(root, f"CeFA/spoof/depth/*.jpg")))
                spoofi = sorted(glob.glob(os.path.join(root, f"CeFA/spoof/ir/*.jpg")))
                
                self.all_liver += liver
                self.all_lived += lived
                self.all_livei += livei
                
                self.all_spoofr += spoofr
                self.all_spoofd += spoofd
                self.all_spoofi += spoofi
            if i == 'S':
                liver = sorted(glob.glob(os.path.join(root, f"SURF/real/profile/*.jpg")))
                lived = sorted(glob.glob(os.path.join(root, f"SURF/real/depth/*.jpg")))
                livei = sorted(glob.glob(os.path.join(root, f"SURF/real/ir/*.jpg")))
                
                spoofr = sorted(glob.glob(os.path.join(root, f"SURF/spoof/profile/*.jpg")))
                spoofd = sorted(glob.glob(os.path.join(root, f"SURF/spoof/depth/*.jpg")))
                spoofi = sorted(glob.glob(os.path.join(root, f"SURF/spoof/ir/*.jpg")))
                
                self.all_liver += liver
                self.all_lived += lived
                self.all_livei += livei
                
                self.all_spoofr += spoofr
                self.all_spoofd += spoofd
                self.all_spoofi += spoofi
            if i == 'W':
                liver = sorted(glob.glob(os.path.join(root, f"WMCA/real/profile/*.jpg")))
                lived = sorted(glob.glob(os.path.join(root, f"WMCA/real/depth/*.jpg")))
                livei = sorted(glob.glob(os.path.join(root, f"WMCA/real/ir/*.jpg")))
                
                spoofr = sorted(glob.glob(os.path.join(root, f"WMCA/spoof/profile/*.jpg")))
                spoofd = sorted(glob.glob(os.path.join(root, f"WMCA/spoof/depth/*.jpg")))
                spoofi = sorted(glob.glob(os.path.join(root, f"WMCA/spoof/ir/*.jpg")))
                
                self.all_liver += liver
                self.all_lived += lived
                self.all_livei += livei
                
                self.all_spoofr += spoofr
                self.all_spoofd += spoofd
                self.all_spoofi += spoofi

            if i == 'intraC':
                # liver = sorted(glob.glob(os.path.join(root, f"test/real/profile/*.jpg")))
                # lived = sorted(glob.glob(os.path.join(root, f"test/real/depth/*.jpg")))
                # livei = sorted(glob.glob(os.path.join(root, f"test/real/ir/*.jpg")))
                
                # spoofr = sorted(glob.glob(os.path.join(root, f"test/spoof/profile/*.jpg")))
                # spoofd = sorted(glob.glob(os.path.join(root, f"test/spoof/depth/*.jpg")))
                # spoofi = sorted(glob.glob(os.path.join(root, f"test/spoof/ir/*.jpg")))
                
                # self.all_liver += liver
                # self.all_lived += lived
                # self.all_livei += livei
                
                # self.all_spoofr += spoofr
                # self.all_spoofd += spoofd
                # self.all_spoofi += spoofi
                self.all_liver = np.load(root + '/test/real/rgb.npy')
                self.all_lived = np.load(root + '/test/real/depth.npy')
                self.all_livei = np.load(root + '/test/real/ir.npy')
                self.all_spoofr= np.load(root + '/test/spoof/rgb.npy')
                self.all_spoofd= np.load(root + '/test/spoof/depth.npy')
                self.all_spoofi= np.load(root + '/test/spoof/ir.npy')
            if i == 'intraS':
                # liver = sorted(glob.glob(os.path.join(root, f"test/real/color/*.jpg")))
                # lived = sorted(glob.glob(os.path.join(root, f"test/real/depth/*.jpg")))
                # livei = sorted(glob.glob(os.path.join(root, f"test/real/ir/*.jpg")))
                
                # spoofr = sorted(glob.glob(os.path.join(root, f"test/spoof/color/*.jpg")))
                # spoofd = sorted(glob.glob(os.path.join(root, f"test/spoof/depth/*.jpg")))
                # spoofi = sorted(glob.glob(os.path.join(root, f"test/spoof/ir/*.jpg")))
                
                # self.all_liver += liver
                # self.all_lived += lived
                # self.all_livei += livei
                
                # self.all_spoofr += spoofr
                # self.all_spoofd += spoofd
                # self.all_spoofi += spoofi

                self.all_liver = np.load(root + '/test/real/rgb.npy')
                self.all_lived = np.load(root + '/test/real/depth.npy')
                self.all_livei = np.load(root + '/test/real/ir.npy')
                self.all_spoofr= np.load(root + '/test/spoof/rgb.npy')
                self.all_spoofd= np.load(root + '/test/spoof/depth.npy')
                self.all_spoofi= np.load(root + '/test/spoof/ir.npy')
                
        # shuffle self.all_liver
        # if train:
        #     random.shuffle(self.all_liver)
        #     random.shuffle(self.all_lived)
        #     random.shuffle(self.all_livei)
        #     random.shuffle(self.all_spoofr)
        #     random.shuffle(self.all_spoofd)
        #     random.shuffle(self.all_spoofi)
        #self.live_labels = np.zeros(len(self.all_liver), dtype=np.int64)
        #self.spoof_labels = np.ones(len(self.all_spoofr), dtype=np.int64)
        self.live_labels = np.ones(len(self.all_liver), dtype=np.int64)
        self.spoof_labels = np.zeros(len(self.all_spoofr), dtype=np.int64)
        self.total_labels = np.concatenate((self.live_labels, self.spoof_labels), axis=0)
        
        
        
        
        self.total_rgb = np.concatenate((self.all_liver, self.all_spoofr), axis=0)
        self.total_depth = np.concatenate((self.all_lived, self.all_spoofd), axis=0)
        self.total_ir = np.concatenate((self.all_livei, self.all_spoofi), axis=0)
        
        self.train = train
        self.size = size
        self.randomcrop = transforms.RandomResizedCrop(self.size)
        
    def transform(self, img1, img2, img3, train = True, size = 224):
        

        # Random crop
        # i, j, h, w = transforms.CenterCrop.get_params(
        #     img1, output_size=(224, 224))
        if train:
            img1 = TF.center_crop(TF.resize(img1, (512,512)), (size,size))
            img2 = TF.center_crop(TF.resize(img2, (512,512)), (size,size))
            img3 = TF.center_crop(TF.resize(img3, (512,512)), (size,size))
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
            img1 = TF.resize(img1, (size,size))
            img2 = TF.resize(img2, (size,size))
            img3 = TF.resize(img3, (size,size))
            
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
        
                
        # rgb_img = torch.stack(get_frame(rgb, self.transform_face)).transpose(0, 1)
        # depth_img = torch.stack(get_frame(depth, self.transform_face)).transpose(0, 1)
        # ir_img = torch.stack(get_frame(ir, self.transform_face)).transpose(0, 1)
        
        if self.protocol[0] == 'inter' or self.protocol[0] == 'intraS' or self.protocol[0] == 'intraC':
            rgb_img = Image.fromarray(rgb)
            depth_img = Image.fromarray(depth)
            ir_img = Image.fromarray(ir)
        else:
            rgb_img = get_frame(rgb)
            depth_img = get_frame(depth)
            ir_img = get_frame(ir)
        
        
        rgb_img, depth_img, ir_img = self.transform(rgb_img, depth_img, ir_img, self.train, self.size)
        
                
        return rgb_img, depth_img, ir_img, labels


    def __len__(self):
        return len(self.total_rgb)


def get_loader( root, protocol, batch_size=10, shuffle=True, train = True, size = 224):
    
    _dataset = FAS_Dataset(root=root,
                           protocol=protocol,
                           train = train,
                           size = size)
    
    return DataLoader(_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


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