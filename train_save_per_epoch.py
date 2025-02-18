import torch
# from models.ViT_Baseline import Multimodality,Mainmodal_Single
from models.ViT import ViT_AvgPool_3modal_CrossAtten_Channel
import torch.optim as optim
import torch.nn as nn
import itertools
import numpy as np
import os
import random
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import logging
from pytz import timezone
from datetime import datetime
import sys
import torchvision.transforms as T
from thop import profile
from balanceloader import *
import warnings
warnings.filterwarnings("ignore")
logging.Formatter.converter = lambda *args: datetime.now(tz=timezone('Asia/Taipei')).timetuple()
from torch.optim.lr_scheduler import StepLR
from pytorch_metric_learning import losses
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def NormalizeData_torch(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))


dataset1 = sys.argv[1]
device_id = 'cuda:0'
results_filename = dataset1.replace('/', '_') + '_MMDG' #_final_version_
results_path = '/shared/shared/yitinglin/PR/' + results_filename

os.system("rm -r "+results_path)
import random 

lr_rate1 = random.choice([7e-5,8e-5])#,0.9,
lr1 = lr_rate1

# batch_size = random.choice([10])
batch_size = 32

scale = random.choice(['None'])

mkdir(results_path)
mkdir('/home/s113062513/PR/logger/')
file_handler = logging.FileHandler(filename='/home/s113062513/PR/logger/'+ results_filename +'_train.log')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
date = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=date, handlers=handlers)

logging.info(f"Batch Size:      {batch_size}")

logging.info("scale: {}".format(scale))

logging.info(f"Train on {dataset1}")

# image shape: torch.Size([3, 256, 256])
root = '/shared/shared/WMCA/grandtest'
if dataset1 == 'train_grandtest':
    root = '/shared/shared/WMCA/grandtest'
if dataset1 == 'train_LOO_glasses':
    root = '/shared/shared/WMCA/LOO_glasses'
if dataset1 == 'train_LOO_rigidmask':
    root = '/shared/shared/WMCA/LOO_rigidmask'
if dataset1 == 'train_LOO_papermask':
    root = '/shared/shared/WMCA/LOO_papermask'
if dataset1 == 'train_LOO_prints':
    root = '/shared/shared/WMCA/LOO_prints'
if dataset1 == 'train_LOO_replay':
    root = '/shared/shared/WMCA/LOO_replay'
if dataset1 == 'train_LOO_flexiblemask':
    root = '/shared/shared/WMCA/LOO_flexiblemask'
if dataset1 == 'train_LOO_fakehead':
    root = '/shared/shared/WMCA/LOO_fakehead'

if dataset1 == 'C' or dataset1 == 'W' or dataset1 == 'S':
    root = '/shared/shared/domain-generalization-multi'
    
# if dataset 1 .split / in front is intraC
if dataset1.split('/')[0] == 'intraC':
    root = '/shared/CeFA_intra/'+dataset1.split('/')[1]
    dataset1 = dataset1.split('/')[0]
# if dataset 1 is intraS
if dataset1 == 'intraS':
    # print('intraS')
    root = '/shared/shared/SURF_intra2/'

Fas_Net = ViT_AvgPool_3modal_CrossAtten_Channel().to(device_id)


criterionCls = nn.CrossEntropyLoss().to(device_id)
cosinloss = nn.CosineSimilarity(dim=1, eps=1e-6).to(device_id)


criterion_mse = nn.MSELoss().to(device_id)
contrastiveloss = losses.NTXentLoss(temperature=0.07)




# Get all parameters
all_parameters = Fas_Net.parameters()
        

optimizerALL        = optim.AdamW(Fas_Net.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)

# optimizerALL        = optim.RAdam(Fas_Net.parameters(), lr=5e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)

Fas_Net.train()

model_save_step = 50
model_save_epoch = 1

save_index = 0

# train_loader = get_loader(root = root, protocol=[dataset1], batch_size=batch_size, shuffle=True, size = 256)
live_loader = get_loader(root = root, protocol=[dataset1], batch_size=int(batch_size*0.5), shuffle=True, size = 224, cls = 'real')
spoof_loader = get_loader(root = root, protocol=[dataset1], batch_size=int(batch_size*0.5), shuffle=True, size = 224, cls = 'spoof')

live_loader_proto = get_loader(root = root, protocol=[dataset1], batch_size=int(1000*0.5), shuffle=True, size = 224, cls = 'real')
spoof_loader_proto = get_loader(root = root, protocol=[dataset1], batch_size=int(1000*0.5), shuffle=True, size = 224, cls = 'spoof')

iternum = max(len(live_loader), len(spoof_loader))
# iternum = len(live_loader)

live_loader = get_inf_iterator(live_loader)
spoof_loader = get_inf_iterator(spoof_loader)

log_step = 50
logging.info(f"iternum={iternum}")



cls = 0.3
sim = 0.7
grad_sizes = {}

spoof_R = None
spoof_I = None
spoof_D = None


def euclidean_distance(a, b):
    return torch.norm(a - b, dim=1)

def ssp_loss(c_m, p_dgt_m, p_dm):
    """
    Compute the SSP loss for a single modality.

    :param c_m: Tensor of shape (batch_size, feature_dim), sample features in modality m
    :param p_dgt_m: Tensor of shape (feature_dim,), prototype of the ground truth domain in modality m
    :param p_dm: List of tensors, each of shape (feature_dim,), prototypes of all domains in modality m
    :return: SSP loss value
    """
    # Compute the Euclidean distance from the sample to the ground truth domain prototype
    numerator = torch.exp(-euclidean_distance(c_m, p_dgt_m))

    # Compute the sum of exponentials of negative Euclidean distances from the sample to all domain prototypes
    denominator = sum(torch.exp(-euclidean_distance(c_m, pd)) for pd in p_dm)

    # Compute the SSP loss
    loss = -torch.log(numerator / denominator)

    return loss.mean()

for epoch in range(50):

    print("Calculating Prototypes...")
    rgb_proto = torch.zeros(2, 768).cuda()
    ir_proto = torch.zeros(2, 768).cuda()
    depth_proto = torch.zeros(2, 768).cuda()
    rgb_proto_prev = torch.zeros(2, 768).cuda()
    ir_proto_prev = torch.zeros(2, 768).cuda()
    depth_proto_prev = torch.zeros(2, 768).cuda()

    momentum = 0.8

    Fas_Net.eval()
    with torch.no_grad():
        live_count = 0
        spoof_count = 0
        for step, sample_batch in enumerate(live_loader_proto):
            rgb_img_live, depth_img_live, ir_img_live, labels_live, atktype_live = sample_batch
            rgb_img_live = rgb_img_live.to(device_id)
            depth_img_live = depth_img_live.to(device_id)
            ir_img_live = ir_img_live.to(device_id)
            labels_live = labels_live.to(device_id)
            atktype_live = atktype_live.to(device_id)

            pred, R, I, D = Fas_Net(rgb_img_live, ir_img_live, depth_img_live)
            #proto_live += torch.stack([F.normalize(R), F.normalize(I), F.normalize(D)])
            rgb_proto[0] += torch.mean(F.normalize(R), 0)
            ir_proto[0] += torch.mean(F.normalize(I), 0)
            depth_proto[0] += torch.mean(F.normalize(D), 0)
            live_count += 1
            
        for step, sample_batch in enumerate(spoof_loader_proto):
            rgb_img_spoof, depth_img_spoof, ir_img_spoof, labels_spoof, atktype_spoof = sample_batch
            rgb_img_spoof = rgb_img_spoof.to(device_id)
            depth_img_spoof = depth_img_spoof.to(device_id)
            ir_img_spoof = ir_img_spoof.to(device_id)
            labels_spoof = labels_spoof.to(device_id)
            atktype_spoof = atktype_spoof.to(device_id)

            pred, R, I, D = Fas_Net(rgb_img_spoof, ir_img_spoof, depth_img_spoof)
            #proto_spoof += torch.stack([F.normalize(R), F.normalize(I), F.normalize(D)])
            rgb_proto[1] += torch.mean(F.normalize(R), 0)
            ir_proto[1] += torch.mean(F.normalize(I), 0)
            depth_proto[1] += torch.mean(F.normalize(D), 0)
            spoof_count += 1

    rgb_proto[0] /= live_count
    ir_proto[0] /= live_count
    depth_proto[0] /= live_count
    rgb_proto[1] /= spoof_count
    ir_proto[1] /= spoof_count
    depth_proto[1] /= spoof_count

    if epoch > 0:
        #proto_live = (1 - momentum) * proto_live_prev + momentum * proto_live
        #proto_spoof = (1 - momentum) * proto_spoof_prev + momentum * proto_spoof
        rgb_proto = (1 - momentum) * rgb_proto_prev + momentum * rgb_proto
        ir_proto = (1 - momentum) * ir_proto_prev + momentum * ir_proto
        depth_proto = (1 - momentum) * depth_proto_prev + momentum * depth_proto

    rgb_proto_prev = rgb_proto
    ir_proto_prev = ir_proto
    depth_proto_prev = depth_proto

    Fas_Net.train()
    for step in range(iternum):

        # ============ one batch extraction ============#
        rgb_img_live, depth_img_live, ir_img_live, labels_live, atktype_live = next(live_loader)
        rgb_img_spoof, depth_img_spoof, ir_img_spoof, labels_spoof, atktype_spoof = next(spoof_loader)
        # ============ one batch extraction ============#
        
        # if step == 0:
        rgb_img = torch.cat([rgb_img_live,rgb_img_spoof], 0).to(device_id)
        depth_img = torch.cat([depth_img_live,depth_img_spoof], 0).to(device_id)
        ir_img = torch.cat([ir_img_live,ir_img_spoof], 0).to(device_id)
        labels = torch.cat([labels_live,labels_spoof], 0).to(device_id)
        atktype = torch.cat([atktype_live,atktype_spoof], 0).to(device_id)
        
        batchidx = list(range(len(rgb_img)))
        random.shuffle(batchidx)

        rgb_img_rand = NormalizeData_torch(rgb_img[batchidx, :])
        depth_img_rand = NormalizeData_torch(depth_img[batchidx, :])
        ir_img_rand = NormalizeData_torch(ir_img[batchidx, :])
        labels_rand = labels[batchidx]
        atktype_rand = atktype[batchidx]
        
        pred, R, I, D\
                                    = Fas_Net((rgb_img_rand)  #).to(device_id)
                                            , (ir_img_rand)
                                            , (depth_img_rand))#.to(device_id)

        Crossentropy = criterionCls(pred, labels_rand)

        R_ssp_loss_live = ssp_loss(F.normalize(R[labels_rand == 1]), rgb_proto[0], rgb_proto)
        I_ssp_loss_live = ssp_loss(F.normalize(I[labels_rand == 1]), ir_proto[0], ir_proto)
        D_ssp_loss_live = ssp_loss(F.normalize(D[labels_rand == 1]), depth_proto[0], depth_proto)
        R_ssp_loss_spoof = ssp_loss(F.normalize(R[labels_rand == 0]), rgb_proto[1], rgb_proto)
        I_ssp_loss_spoof = ssp_loss(F.normalize(I[labels_rand == 0]), ir_proto[1], ir_proto)
        D_ssp_loss_spoof = ssp_loss(F.normalize(D[labels_rand == 0]), depth_proto[1], depth_proto)

        # total_ssp_loss = R_ssp_loss + I_ssp_loss + D_ssp_loss
        total_ssp_loss = R_ssp_loss_live + I_ssp_loss_live + D_ssp_loss_live + R_ssp_loss_spoof + I_ssp_loss_spoof + D_ssp_loss_spoof
        
        TotalLossALL = Crossentropy + 0.05 * total_ssp_loss
        
        optimizerALL.zero_grad()
        TotalLossALL.backward()
        optimizerALL.step()

        if (step + 1) % log_step == 0:
            logging.info('[epoch %d step %d]  Crossentropy: %.4f  ssp_loss: %.4f  TotalLossALL: %.4f'
                         % (epoch + 1, step + 1, Crossentropy.item(), total_ssp_loss.item(), TotalLossALL.item()))
                 
        '''
        if (step + 1) % model_save_step == 0:# and epoch>3:
            mkdir(results_path)
            save_index += 1
            save_dict = Fas_Net.state_dict()

            torch.save(save_dict, os.path.join(results_path,"FASNet-{}.tar".format(save_index)))'''

    if (epoch + 1) % model_save_epoch == 0:    
        mkdir(results_path)
        save_dict = Fas_Net.state_dict()
        torch.save(save_dict, os.path.join(results_path,"FASNet-{}.tar".format(epoch+1)))
            
        


