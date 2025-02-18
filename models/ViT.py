import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch


import math

import torch
import torch.utils.model_zoo as model_zoo
from torch.nn import Parameter
import pdb
import numpy as np
import timm
import timm.models.vision_transformer


# class VisionTransformer(timm.models.vision_transformer.VisionTransformer):

    

#     def forward_features(self, x):
#         x = self.patch_embed(x)
#         cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#         # if self.dist_token is None:
#         x = torch.cat((cls_token, x), dim=1)
#         # else:
#         #     x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
#         x = self.pos_drop(x + self.pos_embed)
#         x = self.blocks(x)
#         # if self.dist_token is None:
#         #     return x[:, 1:]
#         # else:
#         #     return x[:, 2:]

#     def forward(self, x):
#         print(self.forward_features)
#         x = self.forward_features(x)
#         return x


# timm.models.vision_transformer.VisionTransformer = VisionTransformer
        
# class ViT_AvgPool_3modal_CrossAtten_Channel(nn.Module):

#     def __init__(self, pretrained=True):
#         super(ViT_AvgPool_3modal_CrossAtten_Channel, self).__init__()
#         self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)

#         #  binary CE
#         self.fc = nn.Linear(768, 2)
        
#         self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        
#         self.drop = nn.Dropout(0.3)
#         self.drop2d = nn.Dropout2d(0.3)
        
#         # fusion head
#         self.ConvFuse = nn.Sequential(
#             nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(768),
#             nn.ReLU(),    
#         )

#     def forward(self, x1, x2, x3):

#         classtoken1 =  self.vit.forward_features(x1)
#         classtoken2 =  self.vit.forward_features(x2)
#         classtoken3 =  self.vit.forward_features(x3)
        
#         classtoken1 = classtoken1.transpose(1, 2).view(-1, 768, 14, 14)
#         classtoken2 = classtoken2.transpose(1, 2).view(-1, 768, 14, 14) 
#         classtoken3 = classtoken3.transpose(1, 2).view(-1, 768, 14, 14) 
        
#         B,C,H,W = classtoken1.shape
#         h1_temp = classtoken1.view(B,C,-1)
#         h2_temp = classtoken2.view(B,C,-1)
#         h3_temp = classtoken3.view(B,C,-1)
        
#         crossh1_h2 = h2_temp @ h1_temp.transpose(-2, -1)    # [64, 768, 768]
#         #pdb.set_trace()
#         crossh1_h2 =F.softmax(crossh1_h2, dim=-1)
#         crossedh1_h2 = (crossh1_h2 @ h1_temp).contiguous()  # [64, 768, 196]
#         crossedh1_h2 = crossedh1_h2.view(B,C,H,W)
        
#         crossh1_h3 = h3_temp @ h1_temp.transpose(-2, -1)
#         crossh1_h3 =F.softmax(crossh1_h3, dim=-1)
#         crossedh1_h3 = (crossh1_h3 @ h1_temp).contiguous()
#         crossedh1_h3 = crossedh1_h3.view(B,C,H,W)
        
#         #h_concat = torch.cat((classtoken1, crossedh1_h2, crossedh1_h3), dim=1)
#         h_concat = classtoken1 + crossedh1_h2 + crossedh1_h3
#         h_concat = self.ConvFuse(self.drop2d(h_concat))
        
#         regmap8 =  self.avgpool8(h_concat)
        
#         logits = self.fc(self.drop(regmap8.squeeze(-1).squeeze(-1)))
        
#         return logits

class ViT_AvgPool_3modal_CrossAtten_Channel(nn.Module):
    def __init__(self, pretrained=True):
        super(ViT_AvgPool_3modal_CrossAtten_Channel, self).__init__()
        vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0, class_token=True)
        self.class_token = vit.cls_token 
        self.pos_embed = vit.pos_embed
        vit1 = nn.Sequential(*list(vit.children())[:-3])
        self.embed = nn.Sequential(*list(vit1.children())[:-1])
        vit11 = nn.Sequential(*list(vit1.children())[-2]) # Change from -1 to -2
        self.block1_10 = nn.Sequential(*list(vit11.children())[:-1])
        self.block11 = vit11[-1]
        self.block11_up_to_norm2 = nn.Sequential(
            self.block11.norm1,
            self.block11.attn,
            self.block11.ls1,
            self.block11.drop_path1,
        )
        self.block11_norm2 = self.block11.norm2
        self.block11_until_mlp = nn.Sequential(
            self.block11.mlp,
            self.block11.ls2,
            self.block11.drop_path2
        )
        self.ugca = UGCA(channels=768, num_heads=12)  # UGCA instance
        self.uem = UncertaintyEstimationModule(channels=768) 
        self.classifier = nn.Linear(768 * 3, 2)  # Output layer for classification
        # Remaining initializations...

    def forward_features(self, x):
        x = self.embed(x)
        cls_tokens = self.class_token.expand(x.shape[0], -1, -1)  # Duplicate class token for batch
        x = torch.cat((cls_tokens, x), dim=1)  # Prepend class token
        x = x + self.pos_embed  # Add position embedding
        x = self.block1_10(x)
        return x
        
    def forward(self, R, I, D):
        
        # freeze every thing except self.ugca and self.classifier
        for param in self.parameters():
            param.requires_grad = False
        for param in self.ugca.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
        
        # Extract features using ViT
        R = self.forward_features(R)
        I = self.forward_features(I)
        D = self.forward_features(D)
        
        Rx2 = self.block11_up_to_norm2(R)
        Ix2 = self.block11_up_to_norm2(I)
        Dx2 = self.block11_up_to_norm2(D)
        
        Rx3 = self.block11_norm2(Rx2)
        Ix3 = self.block11_norm2(Ix2)
        Dx3 = self.block11_norm2(Dx2)
        
        Rx4 = self.block11_until_mlp(Rx3)
        Ix4 = self.block11_until_mlp(Ix3)
        Dx4 = self.block11_until_mlp(Dx3)
        
        
        # Estimate uncertainty for each modality
        RU = self.uem(Rx2)  
        IU = self.uem(Ix2)
        DU = self.uem(Dx2)

        #fea 1: rgb ; feat2: depth; feat3: ir
        

        # Apply UGCA for cross-modal fusion, guided by uncertainty
        Depth_attention = self.ugca(Rx3, Rx3, Dx3, RU) # Depth attention
        IR_attention = self.ugca(Rx3, Rx3, Ix3, RU) # IR attention
        RGB_attention = self.ugca(Dx3, Dx3, Rx3, DU) # RGB attention
        RGB_attention += self.ugca(Ix3, Ix3, Rx3, IU) # RGB attention
        
        # change the shape of the attention map back to 1, 196, 768
        #change attention map shape to Rx2 shape -1 196 768
        RGB_attention = RGB_attention.reshape(-1, 197, 768)
        IR_attention = IR_attention.reshape(-1, 197, 768)
        Depth_attention = Depth_attention.reshape(-1, 197, 768)
        
        R1 = RGB_attention + Rx3 + Rx4
        I1 = IR_attention + Ix3 + Ix4
        D1 = Depth_attention + Dx3 + Dx4
        
        # extract the class token for each modality
        R = R1[:, 0, :]
        I = I1[:, 0, :]
        D = D1[:, 0, :]

        logits_all = torch.cat([R, I, D], dim=1)
        out_all = self.classifier(logits_all)

        # Continue with processing fused features...
        # Rpred = self.classifier(R)
        # Ipred = self.classifier(I)
        # Dpred = self.classifier(D)
        
        return out_all, R, I, D
        # return out_all, R1, I1, D1
        # return Rpred, Ipred, Dpred, R1, I1, D1
        # return logits  # Output from the model
    
class UAdapter(nn.Module):
    def __init__(self, channels, num_heads):
        super(UAdapter, self).__init__()
        # Assuming the CDC layer is similar to a convolutional layer for the purpose of this example
        self.cdc = Conv2d_cd(channels, channels, kernel_size=3, stride=1, padding=1)
        self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads)
        
    def forward(self, query, key, value):
        # Attention mechanism
        attn_output, _ = self.attention(query, key, value)
        # Central Difference Convolution
        cdc_output = self.cdc(attn_output.transpose(1, 2).view(attn_output.size(0), -1, 14, 14))
        return cdc_output.view(attn_output.size(0), -1, attn_output.size(2)).transpose(1, 2)  # Reshape back to match dimensions


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff
        
class UncertaintyEstimationModule(nn.Module):
    def __init__(self, channels, dropout_rate=0.5):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # Simulate Monte Carlo sampling by applying dropout multiple times
        # This is a simplified version of uncertainty estimation
        dropout_samples = [self.dropout(x) for _ in range(10)]
        sample_stack = torch.stack(dropout_samples)
        uncertainty = torch.var(sample_stack, dim=0)
        return uncertainty
    
class UGCA(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads)
        self.cdc = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        #conv1x1
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        #gelu
        self.gelu = nn.GELU()

    def forward(self, query, key, value, uncertainty):
        # Modulate query with uncertainty
        modulated_query = query * uncertainty
        attn_output, _ = self.attention(modulated_query, key, value)
        # Process attention output with CDC
        # split the token
        token = attn_output[:, 0, :]
        attn_output = attn_output[:, 1:, :]
        cdc_output = self.cdc(attn_output.transpose(1, 2).view(attn_output.size(0), -1, 14, 14))
        conv1x1_output = self.conv1x1(cdc_output)
        # Gelu activation
        cdc_output = self.gelu(conv1x1_output)
        # change cdc_output shape to attn_output shape
        #write this with reshape cdc_output = cdc_output.view(attn_output.size(0), -1, attn_output.size(2)).transpose(1, 2)
        cdc_output = cdc_output.reshape(attn_output.size(0), attn_output.size(2), -1).transpose(1, 2)
        # Add token back to the output
        cdc_output = cdc_output.reshape(-1, 196, 768)
        return torch.cat((token.unsqueeze(1), cdc_output), dim=1)


# main
if __name__ == '__main__':
    # Create a model instance
    model = ViT_AvgPool_3modal_CrossAtten_Channel()
    # Create dummy inputs
    R = torch.randn(1, 3, 224, 224)
    I = R
    D = R
    # Forward pass
    output = model(R, I, D)
    print(output)
    