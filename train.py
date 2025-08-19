import os
import sys
import itertools
import random 
import argparse
import logging
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image
from pytz import timezone
from thop import profile
# from pytorch_metric_learning import losses

from balanceloader import *
from models.ViT import ViT_AvgPool_2modal_CrossAtten_Channel


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def NormalizeData_torch(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

parser = argparse.ArgumentParser(description="config")
parser.add_argument("--train_dataset", type=str, default='CASIA_SURF')
parser.add_argument("--data_root", type=str, default='/media/back/home/chuck/Dataset/CASIA_SURF/challenge/train')
parser.add_argument("--result_path", type=str, default='./saved/')
parser.add_argument("--num_epoch", type=int, default=1)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--weighted_factor", type=float, default=0.05)
parser.add_argument("--learning_rate", type=float, default=5e-5)
args = parser.parse_args()

dataset1 = args.train_dataset
root = args.data_root
results_path = args.result_path
num_epoch = args.num_epoch
patience = args.patience
batch_size = args.batch_size
weighted_factor = args.weighted_factor
learning_rate = args.learning_rate

mkdir(results_path)
device_id = 'cuda:0'

logging.Formatter.converter = lambda *args: datetime.now(tz=timezone('Asia/Taipei')).timetuple()
file_handler = logging.FileHandler(filename='train.log')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
date = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=date, handlers=handlers)
logging.info(f"Batch Size: {batch_size}")
logging.info(f"Weighted Factor: {weighted_factor}")
logging.info(f"Learning Rate: {learning_rate}")
logging.info(f"Train on {dataset1}")

Fas_Net = ViT_AvgPool_2modal_CrossAtten_Channel().to(device_id)

criterionCls = nn.CrossEntropyLoss().to(device_id)
cosinloss = nn.CosineSimilarity(dim=1, eps=1e-6).to(device_id)
criterion_mse = nn.MSELoss().to(device_id)
# contrastiveloss = losses.NTXentLoss(temperature=0.07)

# Get all parameters
all_parameters = Fas_Net.parameters()
        
optimizerALL = optim.AdamW(Fas_Net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)
# optimizerALL = optim.RAdam(Fas_Net.parameters(), lr=5e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)

Fas_Net.train()

model_save_step = 10
model_save_epoch = 1
save_index = 0

live_loader = get_loader(root=root, protocol=['train_list.txt'], batch_size=int(batch_size*0.5), shuffle=True, mode='train', size=256, cls='live')
spoof_loader = get_loader(root=root, protocol=['train_list.txt'], batch_size=int(batch_size*0.5), shuffle=True, mode='train', size=256, cls='spoof')
valid_loader = get_loader(root=root, protocol=['train_list.txt'], batch_size=256, shuffle=False, mode='valid', size=256, cls=None)

live_loader_proto = get_loader(root=root, protocol=['train_list.txt'], batch_size=int(1000*0.5), shuffle=True, mode='train', size=256, cls='live')
spoof_loader_proto = get_loader(root=root, protocol=['train_list.txt'], batch_size=int(1000*0.5), shuffle=True, mode='train', size=256, cls='spoof')

iternum = max(len(live_loader), len(spoof_loader))
# iternum = len(live_loader)

live_loader = get_inf_iterator(live_loader)
spoof_loader = get_inf_iterator(spoof_loader)

log_step = 10
logging.info(f"iternum={iternum}")

cls = 0.3
sim = 0.7
grad_sizes = {}


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

for epoch in range(num_epoch):

    print("Calculating Prototypes...")
    rgb_proto = torch.zeros(2, 320).cuda()
    ir_proto = torch.zeros(2, 320).cuda()
    depth_proto = torch.zeros(2, 320).cuda()
    rgb_proto_prev = torch.zeros(2, 320).cuda()
    ir_proto_prev = torch.zeros(2, 320).cuda()
    depth_proto_prev = torch.zeros(2, 320).cuda()

    momentum = 0.8

    Fas_Net.eval()
    with torch.no_grad():
        live_count = 0
        spoof_count = 0
        for step, sample_batch in enumerate(live_loader_proto):
            rgb_img_live, depth_img_live, ir_img_live, labels_live = sample_batch
            rgb_img_live = rgb_img_live.to(device_id)
            ir_img_live = ir_img_live.to(device_id)
            labels_live = labels_live.to(device_id)

            pred, R, I = Fas_Net(rgb_img_live, ir_img_live)
            rgb_proto[0] += torch.mean(F.normalize(R), 0)
            ir_proto[0] += torch.mean(F.normalize(I), 0)
            live_count += 1
            
        for step, sample_batch in enumerate(spoof_loader_proto):
            rgb_img_spoof, depth_img_spoof, ir_img_spoof, labels_spoof = sample_batch
            rgb_img_spoof = rgb_img_spoof.to(device_id)
            ir_img_spoof = ir_img_spoof.to(device_id)
            labels_spoof = labels_spoof.to(device_id)

            pred, R, I = Fas_Net(rgb_img_spoof, ir_img_spoof)
            rgb_proto[1] += torch.mean(F.normalize(R), 0)
            ir_proto[1] += torch.mean(F.normalize(I), 0)
            spoof_count += 1

    rgb_proto[0] /= live_count
    ir_proto[0] /= live_count
    rgb_proto[1] /= spoof_count
    ir_proto[1] /= spoof_count

    if epoch > 0:
        rgb_proto = (1 - momentum) * rgb_proto_prev + momentum * rgb_proto
        ir_proto = (1 - momentum) * ir_proto_prev + momentum * ir_proto

    rgb_proto_prev = rgb_proto
    ir_proto_prev = ir_proto

    Fas_Net.train()
    for step in range(iternum):

        # ============ one batch extraction ============#
        rgb_img_live, depth_img_live, ir_img_live, labels_live = next(live_loader)
        rgb_img_spoof, depth_img_spoof, ir_img_spoof, labels_spoof = next(spoof_loader)
        # ============ one batch extraction ============#
        
        # if step == 0:
        rgb_img = torch.cat([rgb_img_live, rgb_img_spoof], 0).to(device_id)
        ir_img = torch.cat([ir_img_live, ir_img_spoof], 0).to(device_id)
        labels = torch.cat([labels_live, labels_spoof], 0).to(device_id)
        
        batchidx = list(range(len(rgb_img)))
        random.shuffle(batchidx)

        rgb_img_rand = NormalizeData_torch(rgb_img[batchidx, :])
        ir_img_rand = NormalizeData_torch(ir_img[batchidx, :])
        labels_rand = labels[batchidx]
        
        pred, R, I = Fas_Net(rgb_img_rand, ir_img_rand)

        Crossentropy = criterionCls(pred, labels_rand)

        R_ssp_loss_live = ssp_loss(F.normalize(R[labels_rand == 1]), rgb_proto[0], rgb_proto)
        I_ssp_loss_live = ssp_loss(F.normalize(I[labels_rand == 1]), ir_proto[0], ir_proto)
        R_ssp_loss_spoof = ssp_loss(F.normalize(R[labels_rand == 0]), rgb_proto[1], rgb_proto)
        I_ssp_loss_spoof = ssp_loss(F.normalize(I[labels_rand == 0]), ir_proto[1], ir_proto)

        # total_ssp_loss = R_ssp_loss + I_ssp_loss + D_ssp_loss
        total_ssp_loss = R_ssp_loss_live + I_ssp_loss_live + R_ssp_loss_spoof + I_ssp_loss_spoof
        
        TotalLossALL = Crossentropy + weighted_factor * total_ssp_loss # 0.5 for S->W, 1.0 for C->W, otherwise 0.05
        
        optimizerALL.zero_grad()
        TotalLossALL.backward()
        optimizerALL.step()

        if (step + 1) % log_step == 0:
            logging.info('[epoch %d step %d]  Crossentropy: %.4f  ssp_loss: %.4f  TotalLossALL: %.4f'
                         % (epoch + 1, step + 1, Crossentropy.item(), total_ssp_loss.item(), TotalLossALL.item()))
                 
        
        # if (step + 1) % model_save_step == 0:
        #     save_index += 1
        #     save_dict = Fas_Net.state_dict()

        #     torch.save(save_dict, os.path.join(results_path,"FASNet-{}.pth".format(save_index)))
        #     torch.save(Fas_Net.ugca.state_dict(), os.path.join(results_path,"FASNet-ugca-{}.pth".format(save_index)))
        #     torch.save(Fas_Net.classifier.state_dict(), os.path.join(results_path,"FASNet-classifier-{}.pth".format(save_index)))

    Fas_Net.eval()
    with torch.no_grad():
        live_count = 0
        spoof_count = 0
        acc = 0
        for step, sample_batch in enumerate(valid_loader):
            rgb_img_valid, depth_img_valid, ir_img_valid, labels_valid = sample_batch
            rgb_img_valid = rgb_img_valid.to(device_id)
            ir_img_valid = ir_img_valid.to(device_id)
            labels_valid = labels_valid.to(device_id)

            pred, R, I = Fas_Net(rgb_img_valid, ir_img_valid)
            pred_label = torch.argmax(pred, dim=1)
            acc += (pred_label == labels_valid).sum().item()
            live_count += (labels_valid == 1).sum().item()
            spoof_count += (labels_valid == 0).sum().item()

        logging.info(f"Valid Accuracy: {acc / len(valid_loader.dataset):.4f} ({live_count} live, {spoof_count} spoof)")
        Crossentropy = criterionCls(pred, labels_valid)
        logging.info(f"Valid Crossentropy Loss: {Crossentropy.item():.4f}")
        # check Crossentropy Loss is increasing over patience epochs
        if epoch == 0:
            best_loss = Crossentropy.item()
            best_epoch = epoch
        else:
            if Crossentropy.item() < best_loss:
                best_loss = Crossentropy.item()
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping at epoch {epoch + 1}, best loss: {best_loss:.4f} at epoch {best_epoch + 1}")
                    break

    if (epoch + 1) % model_save_epoch == 0:
        save_index += 1
        torch.save(Fas_Net.state_dict(), os.path.join(results_path, "FASNet-epoch{}.tar".format(epoch+1)))
