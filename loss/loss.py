import torch.nn as nn
import torch
import torch.nn.functional as F
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.losses.generic_pair_loss import GenericPairLoss

cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6).to('cuda')
l2distance = nn.PairwiseDistance(p=2).to('cuda')
l1distance = nn.PairwiseDistance(p=1).to('cuda')

def TripletLossCosine (anchor, positive, negative, margin = 0.8):
    
    # Calculate the cosine similarity
    pos_sim = (torch.mean(cosine_similarity(anchor, positive)))
    neg_sim = (torch.mean(cosine_similarity(anchor, negative)))
    
    # Calculate the loss
    loss = torch.clamp(neg_sim - pos_sim + margin, min=0.0)
    loss = torch.mean(loss)
    
    return loss

def L2ratioloss (point1, point2, point3):
    
    #calculate the distance between the pints
    P12 = torch.mean(l2distance(point1, point2))
    P13 = torch.mean(l2distance(point1, point3))
    P23 = torch.mean(l2distance(point2, point3))
    
    #normalized p12 p13 and p23
    NP12 = P12 / (P12 + P13 + P23)
    NP13 = P13 / (P12 + P13 + P23)
    # NP23 = P23 / (P12 + P13 + P23)
    
    total_loss = 0
    
    for i in range(len(point1)):
        
        #calculate the distance between points at the same index
        p12 = torch.mean(l2distance(point1[i], point2[i]))
        p13 = torch.mean(l2distance(point1[i], point3[i]))
        p23 = torch.mean(l2distance(point2[i], point3[i]))
        
        #normalized p12 p13 and p23
        np12 = p12 / (p12 + p13 + p23)
        np13 = p13 / (p12 + p13 + p23)
        # np23 = p23 / (p12 + p13 + p23)
        
        #calculate the loss between NP12 NP13 and np12 np13 for them to be equal and this loss is added to the total loss
        loss = torch.abs(NP12 - np12) + torch.abs(NP13 - np13)
        total_loss += loss
        
    #calculate the mean of the total loss
    total_loss = total_loss / len(point1)
    
    return total_loss

def brownian_bridge_loss(Z_V, Z_T, Z_a_V, t=0.25):
    N = Z_V.size(0) # Assuming the batch size is the first dimension
    loss = 0
    
    for j in range(N):
        # inner_product_V_aV = torch.inner(Z_V[j], Z_a_V[j])
        # inner_product_T_aV = torch.inner(Z_T[j], Z_a_V[j])
        # inner_product_V_T = torch.inner(Z_V[j], Z_T[j])

        # Compute the numerator and denominator
        numerator = t[j] * torch.inner(Z_V[j], Z_a_V[j]) + (1 - t[j]) * torch.inner(Z_T[j], Z_a_V[j])
        denominator = t[j]**2 + (1 - t[j])**2 + 2 * t[j] * (1 - t[j]) * torch.inner(Z_V[j], Z_T[j])

        # Compute the individual terms of the loss and add to the total loss
        loss += numerator / denominator

    # Divide by N to average over the batch size
    loss = loss / N

    return torch.mean(loss)
    
def livespoof_similarity_loss (rgb,ir,depth,Rproto,Iproto,Dproto):
    rgb,ir,depth = F.normalize(rgb, p=2, dim=1), F.normalize(ir, p=2, dim=1), F.normalize(depth, p=2, dim=1)
    Rproto, Iproto, Dproto = F.normalize(Rproto, p=2, dim=1), F.normalize(Iproto, p=2, dim=1), F.normalize(Dproto, p=2, dim=1)
    
    loss=0
    count=0
    for i in range(len(rgb)):
        r_l = torch.abs(F.cosine_similarity(rgb[i], Rproto[0], dim=0))
        i_l = torch.abs(F.cosine_similarity(ir[i], Iproto[0], dim=0))
        d_l = torch.abs(F.cosine_similarity(depth[i], Dproto[0], dim=0))
        
        loss += (r_l + i_l + d_l)
        count+=1
    return loss/count

def livespoof_feature_cross_loss (rgb,ir,depth,Rproto,Iproto,Dproto):
    rgb,ir,depth = F.normalize(rgb, p=2, dim=1), F.normalize(ir, p=2, dim=1), F.normalize(depth, p=2, dim=1)
    Rproto, Iproto, Dproto = F.normalize(Rproto, p=2, dim=1), F.normalize(Iproto, p=2, dim=1), F.normalize(Dproto, p=2, dim=1)

    r_i = ir - rgb
    r_d = rgb - depth
    d_i = depth - ir

    r_i_p = Iproto - Rproto
    r_d_p = Rproto - Dproto
    d_i_p = Dproto - Iproto
    
    loss = 0
    count=0
    for i in range(len(rgb)):
        s_r_i = torch.abs(F.cosine_similarity(r_i[i], r_i_p[0], dim=0))
        s_r_d = torch.abs(F.cosine_similarity(r_d[i], r_d_p[0], dim=0))
        s_d_i = torch.abs(F.cosine_similarity(d_i[i], d_i_p[0], dim=0))
        
        loss += (s_r_i + s_r_d + s_d_i)
        count+=1
    return loss/count

def live_feature_cross_loss (rgb,ir,depth):
    # r_i = (cosine_similarity(rgb, ir))
    # r_d = (cosine_similarity(rgb, depth))
    # d_i = (cosine_similarity(depth, ir))
    ir = F.normalize(ir, p=2, dim=1)
    depth = F.normalize(depth, p=2, dim=1)
    rgb = F.normalize(rgb, p=2, dim=1)
    r_i = ir - rgb
    r_d = rgb - depth
    d_i = depth - ir
    
    s_r_i = torch.empty(len(r_i)*len(r_i)-len(r_i), dtype=torch.float32)
    s_r_d = torch.empty(len(r_i)*len(r_i)-len(r_i), dtype=torch.float32)
    s_d_i = torch.empty(len(r_i)*len(r_i)-len(r_i), dtype=torch.float32)
    count = 0
    loss = 0
    for i in range(len(r_i)):
        for j in range(len(r_i)):
            if i == j:
                continue
            s_r_i[count] = 1 - F.cosine_similarity((r_i[i]), r_i[j], dim=0)
            s_r_d[count] = 1 - F.cosine_similarity((r_d[i]), r_d[j], dim=0)
            s_d_i[count] = 1 - F.cosine_similarity((d_i[i]), d_i[j], dim=0)
            
            count+=1
    
    return torch.mean(s_r_i) + torch.mean(s_r_d) + torch.mean(s_d_i)


def triag_test(Rproto,Iproto,Dproto,rgb,ir,depth,alpha = 0.5):
    # flatten the features
    Rproto, Iproto, Dproto = Rproto.view(Rproto.size(0), -1), Iproto.view(Iproto.size(0), -1), Dproto.view(Dproto.size(0), -1)
    rgb, ir, depth = rgb.view(rgb.size(0), -1), ir.view(ir.size(0), -1), depth.view(depth.size(0), -1)
    
    
    Rproto, Iproto, Dproto = F.normalize(Rproto, p=2, dim=1), F.normalize(Iproto, p=2, dim=1), F.normalize(Dproto, p=2, dim=1)
    rgb, ir, depth = F.normalize(rgb, p=2, dim=1), F.normalize(ir, p=2, dim=1), F.normalize(depth, p=2, dim=1)
    # mergedproto = torch.cat((Rproto, Iproto, Dproto), dim=0)
    # mergedproto = F.normalize(mergedproto, p=2, dim=1)
    # Rproto, Iproto, Dproto = mergedproto[:len(Rproto)], mergedproto[len(Rproto):len(Rproto)+len(Iproto)], mergedproto[len(Rproto)+len(Iproto):]
    # merged = torch.cat((rgb, ir, depth), dim=0)
    # merged = F.normalize(merged, p=2, dim=1)
    # rgb, ir, depth = merged[:len(rgb)], merged[len(rgb):len(rgb)+len(ir)], merged[len(rgb)+len(ir):]
    
    r_i, r_d, d_i = rgb - ir, depth - rgb, ir - depth
    r_i_p, r_d_p, d_i_p = Rproto - Iproto, Dproto - Rproto, Iproto - Dproto
    
    s_r_i = torch.empty(len(r_i), dtype=torch.float32)
    s_r_d = torch.empty(len(r_i), dtype=torch.float32)
    s_d_i = torch.empty(len(r_i), dtype=torch.float32)
    
    d_r_i = torch.empty(len(r_i), dtype=torch.float32)
    d_r_d = torch.empty(len(r_i), dtype=torch.float32)
    d_d_i = torch.empty(len(r_i), dtype=torch.float32)
    
    for i in range(len(r_i)):
        s_r_i[i] = 1 - F.cosine_similarity(r_i[i], r_i_p[0], dim=0)
        s_r_d[i] = 1 - F.cosine_similarity(r_d[i], r_d_p[0], dim=0)
        s_d_i[i] = 1 - F.cosine_similarity(d_i[i], d_i_p[0], dim=0)
        
        #calculate distance between the points
        d_r_i[i] = (l2distance(r_i[i], r_i_p[0]))
        d_r_d[i] = (l2distance(r_d[i], r_d_p[0]))
        d_d_i[i] = (l2distance(d_i[i], d_i_p[0])) 
    return (s_r_i + s_r_d + s_d_i) *alpha + (d_r_i + d_r_d + d_d_i) *(1-alpha)


    
def cosine_losses(dr1, ir1, rr1, dr2, ir2, rr2, dr3, ir3, rr3):
    aR2I1 = torch.mean(cosine_similarity(rr1, ir1)).abs()
    aR2D1 = torch.mean(cosine_similarity(rr1, dr1)).abs()
    aI2D1 = torch.mean(cosine_similarity(ir1, dr1)).abs()

    aR2I2 = torch.mean(cosine_similarity(rr2, ir2))-0.1
    aR2D2 = torch.mean(cosine_similarity(rr2, dr2))-0.1
    aI2D2 = torch.mean(cosine_similarity(ir2, dr2))-0.1
    
    aR2I21 = torch.mean(cosine_similarity(rr2, ir2))
    aR2D21 = torch.mean(cosine_similarity(rr2, dr2))
    aI2D21 = torch.mean(cosine_similarity(ir2, dr2))
    
    if  aR2I2 < 0:
        aR2I2 =  aR2I2.abs()# - aR2I2.abs()
    if  aR2D2 < 0:
        aR2D2 =  aR2D2.abs()# - aR2D2.abs()
    if  aI2D2 < 0:
        aI2D2 =  aI2D2.abs()# - aI2D2.abs()

    aR2I3 = torch.mean(cosine_similarity(rr3, ir3))-0.1
    aR2D3 = torch.mean(cosine_similarity(rr3, dr3))-0.1
    aI2D3 = torch.mean(cosine_similarity(ir3, dr3))-0.1
    
    aR2I31 = torch.mean(cosine_similarity(rr3, ir3))
    aR2D31 = torch.mean(cosine_similarity(rr3, dr3))
    aI2D31 = torch.mean(cosine_similarity(ir3, dr3))
    
    if  aR2I3 < 0:
        aR2I3 =  aR2I3.abs() - aR2I3.abs()
    if  aR2D3 < 0:
        aR2D3 =  aR2D3.abs() - aR2D3.abs()
    if  aI2D3 < 0:
        aI2D3 =  aI2D3.abs() - aI2D3.abs()
    return aR2I1, aR2D1, aI2D1, aR2I2, aR2D2, aI2D2, aR2I3, aR2D3, aI2D3, aR2I21, aR2D21, aI2D21, aR2I31, aR2D31, aI2D31
    
    
def cosine_losses1(dr3, ir3, rr3):

    # rr3, ir3, dr3 = F.normalize(rr3), F.normalize(ir3), F.normalize(dr3)
    rr3, ir3, dr3 = F.normalize(rr3, p=2, dim=1), F.normalize(ir3, p=2, dim=1), F.normalize(dr3, p=2, dim=1)
    # normalize the features together
    
    aR2I3 = torch.abs(torch.mean(cosine_similarity(rr3, ir3)))
    aR2D3 = torch.abs(torch.mean(cosine_similarity(rr3, dr3)))
    aI2D3 = torch.abs(torch.mean(cosine_similarity(ir3, dr3)))
    if aR2I3 < 0:
        # aR2I3 will be 0
        aR2I3 = 0
    if aR2D3 < 0:
        # aR2D3 will be 0
        aR2D3 = 0
    if aI2D3 < 0:
        # aI2D3 will be 0
        aI2D3 = 0
    

    return (aR2I3+aR2D3+aI2D3)

def cosine_losses2(f, f1, f2, f3):

    # f, f1, f2, f3 = F.normalize(f), F.normalize(f1), F.normalize(f2), F.normalize(f3)

    aR2I3 = torch.abs(torch.mean(cosine_similarity(f, f1)))
    aR2D3 = torch.abs(torch.mean(cosine_similarity(f, f2)))
    aI2D3 = torch.abs(torch.mean(cosine_similarity(f, f3)))
    if aR2I3 < 0:
        # aR2I3 will be 0
        aR2I3 = 0
    if aR2D3 < 0:
        # aR2D3 will be 0
        aR2D3 = 0
    if aI2D3 < 0:
        # aI2D3 will be 0
        aI2D3 = 0
    

    return (aR2I3+aR2D3+aI2D3)





    
def NTXentLoss(features, labels, temperature=0.07, device='cuda', lower_temp = 0.07):
    """
    Compute the NTXent Loss.
    
    Args:
        features (torch.Tensor): Feature representations with shape (batch_size, feature_dim).
        labels (torch.Tensor): Ground truth labels corresponding to the features.
        temperature (float): Temperature parameter to scale the dot products.
        device (str): Device for tensor operations.
        
    Returns:
        torch.Tensor: The computed loss value.
    """
    batch_size = features.size(0)
    
    # Normalize the features
    features = F.normalize(features, p=2, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.mm(features, features.t())
    
    # Create masks for positive and negative samples
    labels = labels.unsqueeze(1)
    # positive_mask = (labels == labels.t()).float().to(device)
    # negative_mask = (1 - positive_mask).to(device)
    
    labels_zero = (labels == 0).float().to(device)
    positive_mask = ((labels == labels.t()).float().to(device)* labels_zero.unsqueeze(0))
    

    at_least_one_zero_mask = labels_zero.unsqueeze(0).mT + labels_zero.unsqueeze(0)
    at_least_one_zero_mask = (at_least_one_zero_mask >= 1).float()
    negative_mask = (1 - positive_mask).to(device) * at_least_one_zero_mask 
    
    labels_one = (labels == 1).float().to(device)
    mask_one = (labels_one.unsqueeze(0).mT + labels_one.unsqueeze(0))*1
    labels_2_3 = (labels == 2).float().to(device) + (labels == 3).float().to(device)
    mask_2_3 = (labels_2_3.unsqueeze(0).mT + labels_2_3.unsqueeze(0))*1
    
    weighted_mask = mask_one + mask_2_3
    negative_mask = negative_mask * weighted_mask
    for i in range(len(positive_mask[0])):
        for j in range(len(positive_mask[0][i])):
            if i == j:
                positive_mask[0][i][j] = 0
    # Handle numerical stability
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = (similarity_matrix - logits_max.detach())
    # Extract logits for positive and negative samples
    logits_pos = torch.masked_select(logits, positive_mask.bool())
    logits_neg = torch.masked_select(logits, negative_mask.bool())
    
    
    logits_neg_mask = torch.masked_select(negative_mask, negative_mask.bool())
    logits_neg = logits_neg * logits_neg_mask
    
    # numerator = torch.exp(logits_pos)
    # denominator = torch.sum(torch.exp(logits_neg), dim=0) + numerator
    # log_exp = torch.log((numerator / denominator))
    # return -torch.mean(log_exp)
    
    # Remove the self-positives (diagonal entries)
    numerator = torch.exp(logits_pos / temperature)
    denominator = torch.sum(torch.exp(logits_neg / lower_temp)) + numerator
    

    # Mean of the negative log likelihood
    log_prob = torch.log(numerator) - torch.log(denominator)
    
    return -torch.mean(log_prob)

def NTXentLoss1(features, labels, temperature=0.1, device='cuda', scale = 0.1):
    """
    Compute the NTXent Loss.
    
    Args:
        features (torch.Tensor): Feature representations with shape (batch_size, feature_dim).
        labels (torch.Tensor): Ground truth labels corresponding to the features.
        temperature (float): Temperature parameter to scale the dot products.
        device (str): Device for tensor operations.
        
    Returns:
        torch.Tensor: The computed loss value.
    """
    batch_size = features.size(0)
    
    features = F.normalize(features, p=2, dim=1)
    
    similarity_matrix = torch.mm(features, features.t())
    
    labels = labels.unsqueeze(1)

    
    #0 live, 1 & 2 live diff modality, 3 ~ spoof own class
    positive_labels_mask = torch.zeros(labels.shape).to(device)
    max_labels = torch.max(labels).to(dtype=torch.long, device=device)

    for i in range(3, max_labels+1):
        positive_labels_mask += (labels == i).float().to(device)
    labels_zero = (labels == 0).float().to(device)
    positive_mask = ((labels == labels.t()).float().to(device)* positive_labels_mask.unsqueeze(0))

    

    at_least_one_zero_mask = positive_labels_mask.unsqueeze(0).mT + positive_labels_mask.unsqueeze(0)
    at_least_one_zero_mask = (at_least_one_zero_mask >= 1).float()
    negative_mask = (1 - positive_mask).to(device) * at_least_one_zero_mask 


    labels_zero = (labels == 0).float().to(device)
    mask_zero = (labels_zero.unsqueeze(0).mT + labels_zero.unsqueeze(0))*0.9
    labels_1_2 = (labels == 1).float().to(device) + (labels == 2).float().to(device)
    mask_1_2 = (labels_1_2.unsqueeze(0).mT + labels_1_2.unsqueeze(0))*1

    # create a labels_non where labels greater than 2
    labels_non = (labels >= 3).float().to(device)
    mask_non = (labels_non.unsqueeze(0).mT + labels_non.unsqueeze(0))*scale - scale

    
    weighted_mask = mask_zero + mask_1_2 + mask_non
    negative_mask = negative_mask * weighted_mask

    # Handle numerical stability
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = (similarity_matrix - logits_max.detach())
    
    # Extract logits for positive and negative samples
    logits_pos = torch.masked_select(logits, positive_mask.bool())
    logits_neg = torch.masked_select(logits, negative_mask.bool())
    
    
    logits_neg_mask = torch.masked_select(negative_mask, negative_mask.bool())
    logits_neg = logits_neg * logits_neg_mask
    
    # numerator = torch.exp(logits_pos)
    # denominator = torch.sum(torch.exp(logits_neg), dim=0) + numerator
    # log_exp = torch.log((numerator / denominator))
    # return -torch.mean(log_exp)
    
    # Remove the self-positives (diagonal entries)
    numerator = torch.exp(logits_pos / temperature)
    denominator = torch.sum(torch.exp(logits_neg / temperature)) + numerator
    


    # Mean of the negative log likelihood
    log_prob = torch.log(numerator) - torch.log(denominator)
    
    return -torch.mean(log_prob)
    
feature1 = torch.randn(8, 64, requires_grad=True)
labels = torch.tensor([0, 0, 5,1,2, 2, 3,5])
# labels1 = torch.tensor([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])
# logits_pos = torch.masked_select(labels, labels1.bool())
loss = NTXentLoss1(feature1, labels, device='cpu')
    
    
    