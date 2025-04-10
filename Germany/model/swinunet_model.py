import copy
import logging
import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from model.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys,SwinTransformerSys_Two
import torch.nn.functional as F
# from model.TSViT_module import ChannelAttention
from model.CBAM import CBAMBlock



class SwinUnet(nn.Module):
    def __init__(self, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.swin_unet = SwinTransformerSys(img_size=64,
                                patch_size=4,
                                in_chans=143,
                                num_classes=3,
                                embed_dim=96,
                                depths=[ 2, 2, 2, 2 ],
                                num_heads=[ 3, 6, 12, 24 ],
                                window_size=8,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.0,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)

    def forward(self, x, target_time=None):
        # x = torch.flatten(x, start_dim=1, end_dim=2)
        # # print(x.shape)
        # target_size = (64, 64) # 224-7
        # x = torch.nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=False)       
        
        # print(x.shape)
        # if x.size()[1] == 1:
        #     x = x.repeat(1,3,1,1)
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        logits = self.swin_unet(x)
        # target_size = (50, 65)
        # logits = torch.nn.functional.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)  
        return logits
    
class SwinUnet_CAM(nn.Module):
    def __init__(self, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet_CAM, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.ca = ChannelAttention(in_planes=143, ratio=16)
        self.swin_unet = SwinTransformerSys(img_size=64,
                                patch_size=4,
                                in_chans=143,
                                num_classes=3,
                                embed_dim=96,
                                depths=[ 2, 2, 2, 2 ],
                                num_heads=[ 3, 6, 12, 24 ],
                                window_size=8,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.0,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)

    def forward(self, x, target_time=None):
        B, T, C, H, W = x.shape
        # x = torch.flatten(x, start_dim=1, end_dim=2)
        # # print(x.shape)
        # target_size = (64, 64) # 224-7
        # x = torch.nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=False)       
        
        # print(x.shape)
        # if x.size()[1] == 1:
        #     x = x.repeat(1,3,1,1)
        
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        residual = x
        out = self.ca(x)
        x_cam = x*out + residual       
        x = rearrange(x_cam, 'b (t c) h w -> b t c h w', t=T)       
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        logits = self.swin_unet(x)
        # target_size = (50, 65)
        # logits = torch.nn.functional.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)  
        return logits    
    
# class SwinUnet_Two(nn.Module):
#     def __init__(self, img_size=224, num_classes=21843, zero_head=False, vis=False):
#         super(SwinUnet_Two, self).__init__()
#         self.num_classes = num_classes
#         self.zero_head = zero_head
#         self.swin_unet = SwinTransformerSys_Two(img_size=64,
#                                 patch_size=4,
#                                 in_chans=12,
#                                 num_classes=3,
#                                 embed_dim=96,
#                                 depths=[ 2, 2, 2, 2 ],
#                                 num_heads=[ 3, 6, 12, 24 ],
#                                 window_size=8,
#                                 mlp_ratio=4.,
#                                 qkv_bias=True,
#                                 qk_scale=None,
#                                 drop_rate=0.0,
#                                 drop_path_rate=0.0,
#                                 ape=False,
#                                 patch_norm=True,
#                                 use_checkpoint=False)

#     def forward(self, x, target_time=None):
#         B, T, C, H, W = x.shape
#         original_height, original_width = H, W
#         target_height, target_width  = 64, 64
           
#         # For Padding             
#         pad_height = target_height - original_height
#         pad_width = target_width - original_width
#         top = pad_height // 2
#         bottom = pad_height - top
#         left = pad_width // 2
#         right = pad_width - left
        
#         x = F.pad(x, (left, right, top, bottom), 'constant', 0)        
#         x = rearrange(x, 'b t c h w -> (b t) c h w')

             
#         if x.size()[1] == 1:
#             x = x.repeat(1,3,1,1)
            
#         logits1,logits2 = self.swin_unet(x)
        
#         # For Recover Index
#         original_height, original_width = 64, 64
#         target_height, target_width  = 50, 65
                   
#         # For Padding             
#         pad_height = target_height - original_height
#         pad_width = target_width - original_width
#         top = pad_height // 2
#         bottom = pad_height - top
#         left = pad_width // 2
#         right = pad_width - left
        
#         logits1 = F.pad(logits1, (left, right, top, bottom), 'constant', 0)  
#         logits2 = F.pad(logits2, (left, right, top, bottom), 'constant', 0)
#         logits1 = rearrange(logits1, '(b t) c h w -> b t c h w', c = 3, t = 6).mean(dim=1)
#         logits2 = rearrange(logits2, '(b t) c h w -> b t c h w', c = 1, t = 6).mean(dim=1)       
#         return logits1,logits2

   
# class SwinUnet_CAM_Two(nn.Module):
#     def __init__(self, img_size=224, num_classes=21843, zero_head=False, vis=False):
#         super(SwinUnet_CAM_Two, self).__init__()
#         self.num_classes = num_classes
#         self.zero_head = zero_head
#         self.ca = ChannelAttention(in_planes=1*143, ratio=16)         
#         self.swin_unet = SwinTransformerSys_Two(img_size=64,
#                                 patch_size=4,
#                                 in_chans=143,
#                                 num_classes=3,
#                                 embed_dim=96,
#                                 depths=[ 2, 2, 2, 2 ],
#                                 num_heads=[ 3, 6, 12, 24 ],
#                                 window_size=8,
#                                 mlp_ratio=4.,
#                                 qkv_bias=True,
#                                 qk_scale=None,
#                                 drop_rate=0.0,
#                                 drop_path_rate=0.0,
#                                 ape=False,
#                                 patch_norm=True,
#                                 use_checkpoint=False)

#     def forward(self, x, target_time=None):
#         B, T, C, H, W = x.shape
#         # print(x.shape)
#         # original_height, original_width = H, W
#         # target_height, target_width  = 64, 64
           
#         # For Padding             
#         # pad_height = target_height - original_height
#         # pad_width = target_width - original_width
#         # top = pad_height // 2
#         # bottom = pad_height - top
#         # left = pad_width // 2
#         # right = pad_width - left
        
#         # x = F.pad(x, (left, right, top, bottom), 'constant', 0) 
        
#         x = rearrange(x, 'b t c h w -> b (t c) h w')
#         residual = x
#         out = self.ca(x)
#         x_cam = x*out + residual       
#         x = rearrange(x_cam, 'b (t c) h w -> b t c h w', t=T)
        
#         x = rearrange(x, 'b t c h w -> (b t) c h w')
             
#         if x.size()[1] == 1:
#             x = x.repeat(1,3,1,1)
            
#         logits1,logits2 = self.swin_unet(x)
        
#         # # For Recover Index
#         # original_height, original_width = 64, 64
#         # target_height, target_width  = 50, 65
                   
#         # # For Padding             
#         # pad_height = target_height - original_height
#         # pad_width = target_width - original_width
#         # top = pad_height // 2
#         # bottom = pad_height - top
#         # left = pad_width // 2
#         # right = pad_width - left
        
#         # logits1 = F.pad(logits1, (left, right, top, bottom), 'constant', 0)  
#         # logits2 = F.pad(logits2, (left, right, top, bottom), 'constant', 0)
#         logits1 = rearrange(logits1, '(b t) c h w -> b t c h w', c = 3, t = T).mean(dim=1)
#         logits2 = rearrange(logits2, '(b t) c h w -> b t c h w', c = 1, t = T).mean(dim=1)       
#         return logits1,logits2

class SwinUnet_Two(nn.Module):
    def __init__(self, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet_Two, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.swin_unet = SwinTransformerSys_Two(img_size=64,
                                patch_size=4,
                                in_chans=12,
                                num_classes=3,
                                embed_dim=96,
                                depths=[ 2, 2, 2, 2 ],
                                num_heads=[ 3, 6, 12, 24 ],
                                window_size=8,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.0,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)

    def forward(self, x, target_time=None):
        B, T, C, H, W = x.shape
        original_height, original_width = H, W
        target_height, target_width  = 64, 64
           
        # For Padding             
        pad_height = target_height - original_height
        pad_width = target_width - original_width
        top = pad_height // 2
        bottom = pad_height - top
        left = pad_width // 2
        right = pad_width - left
        
        x = F.pad(x, (left, right, top, bottom), 'constant', 0)        
        x = rearrange(x, 'b t c h w -> (b t) c h w')

             
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
            
        logits1,logits2 = self.swin_unet(x)
        
        # For Recover Index
        original_height, original_width = 64, 64
        target_height, target_width  = 50, 65
                   
        # For Padding             
        pad_height = target_height - original_height
        pad_width = target_width - original_width
        top = pad_height // 2
        bottom = pad_height - top
        left = pad_width // 2
        right = pad_width - left
        
        logits1 = F.pad(logits1, (left, right, top, bottom), 'constant', 0)  
        logits2 = F.pad(logits2, (left, right, top, bottom), 'constant', 0)
        logits1 = rearrange(logits1, '(b t) c h w -> b t c h w', c = 3, t = T).mean(dim=1)
        logits2 = rearrange(logits2, '(b t) c h w -> b t c h w', c = 1, t = T).mean(dim=1)       
        return logits1,logits2

   
# class SwinUnet_CAM_Two(nn.Module):
#     def __init__(self, img_size=224, num_classes=21843, zero_head=False, vis=False):
#         super(SwinUnet_CAM_Two, self).__init__()
#         self.num_classes = num_classes
#         self.zero_head = zero_head
#         self.ca = ChannelAttention(in_planes=143, ratio=16)         
#         self.swin_unet = SwinTransformerSys_Two(img_size=64,
#                                 patch_size=4,
#                                 in_chans=143,
#                                 num_classes=3,
#                                 embed_dim=96,
#                                 depths=[ 2, 2, 2, 2 ],
#                                 num_heads=[ 3, 6, 12, 24 ],
#                                 window_size=8,
#                                 mlp_ratio=4.,
#                                 qkv_bias=True,
#                                 qk_scale=None,
#                                 drop_rate=0.0,
#                                 drop_path_rate=0.0,
#                                 ape=False,
#                                 patch_norm=True,
#                                 use_checkpoint=False)

#     def forward(self, x, target_time=None):
#         B, T, C, H, W = x.shape
#         # print(x.shape)
#         # original_height, original_width = H, W
#         # target_height, target_width  = 64, 64
           
#         # # For Padding             
#         # pad_height = target_height - original_height
#         # pad_width = target_width - original_width
#         # top = pad_height // 2
#         # bottom = pad_height - top
#         # left = pad_width // 2
#         # right = pad_width - left
        
#         # x = F.pad(x, (left, right, top, bottom), 'constant', 0) 
        
#         x = rearrange(x, 'b t c h w -> b (t c) h w')
#         residual = x
#         out = self.ca(x)
#         x_cam = x*out + residual       
#         x = rearrange(x_cam, 'b (t c) h w -> b t c h w', t=T)
        
#         x = rearrange(x, 'b t c h w -> (b t) c h w')
             
#         if x.size()[1] == 1:
#             x = x.repeat(1,3,1,1)
            
#         logits1,logits2 = self.swin_unet(x)
        
#         # For Recover Index
#         # original_height, original_width = 64, 64
#         # target_height, target_width  = 50, 65
                   
#         # # For Padding             
#         # pad_height = target_height - original_height
#         # pad_width = target_width - original_width
#         # top = pad_height // 2
#         # bottom = pad_height - top
#         # left = pad_width // 2
#         # right = pad_width - left
        
#         # logits1 = F.pad(logits1, (left, right, top, bottom), 'constant', 0)  
#         # logits2 = F.pad(logits2, (left, right, top, bottom), 'constant', 0)
#         logits1 = rearrange(logits1, '(b t) c h w -> b t c h w', c = 3, t = T).mean(dim=1)
#         logits2 = rearrange(logits2, '(b t) c h w -> b t c h w', c = 1, t = T).mean(dim=1)       
#         return logits1,logits2

class FPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPNBlock, self).__init__()
        
        # Convolution layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Skip connection layer to match channels
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # Match skip connection to output channels
        
        # Upsample layer to match spatial sizes
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Batch normalization and ReLU
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Final convolution to downsample to 64x64
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, skip):
        # First, upsample the input tensor `x`
        x = self.upsample(x)
        # print(f"Shape of x: {x.shape}")
        # print(f"Shape of skip: {skip.shape}")

        
        # Check if the spatial size of `x` and `skip` match, if not, adjust it
        if x.shape[2:] != skip.shape[2:]:
            skip = nn.functional.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        # Ensure skip connection has the same channel depth as the output tensor (through 1x1 convolution)
        skip = self.skip_conv(skip)  # Apply 1x1 convolution to match the channels
        
        # Now, add the `x` and `skip` tensors (they should have the same channel size)
        x = self.conv1(x) + skip  # Fusion with skip connection
        
        # Further processing
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)

        # Downsample using convolution to reduce size to 64x64
        x = self.final_conv(x)

        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torchvision.models.video as video_models
from torchvision import models

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.models.video import r3d_18

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# import video_models  # Assuming your ResNet3D model is here
# from CBAM import CBAMBlock  # Assuming CBAM is defined
# from fpn import FPNBlock  # Assuming FPNBlock is defined
# from swin_transformer import SwinTransformerSys_Two  # Assuming Swin Transformer is defined

# 


    
# Assuming CoordAttention is defined elsewhere as CoordAttention
class SwinUnet_CAM_Two(nn.Module):
    def __init__(self, img_size=224, num_classes=21843, num_frames=8, zero_head=False, vis=False):
        super(SwinUnet_CAM_Two, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.num_frames = num_frames

        # Channel Attention (CBAM)
        self.ca = CBAMBlock(channel=143)  # Assuming CBAMBlock is defined elsewhere

        # Load EfficientNet-B4 as the backbone (for feature extraction)
        self.efficientnet = models.efficientnet_b4(pretrained=True)

        # Modify the first Conv2d layer to accept 143 input channels
        self.efficientnet.features[0][0] = nn.Conv2d(143, 48, kernel_size=3, stride=2, padding=1, bias=False)

        # Modify the final feature layer to output 143 channels
        last_channel = self.efficientnet.features[-1][0].out_channels  # Last feature layer output channels
        self.output_conv = nn.Conv2d(last_channel, 143, kernel_size=1, stride=1, padding=0, bias=False)

        # Use EfficientNet features only (no classification head)
        self.features = self.efficientnet.features

        self.fpn1 = FPNBlock(143, 96)  # Updated input channels for FPN
        self.fpn2 = FPNBlock(96, 64)
        self.fpn3 = FPNBlock(64, 32)

        # CoordAttention
        # Assuming CoordAttention is implemented correctly

        # Your Swin Unet (after CoordAttention)
        self.swin_unet = SwinTransformerSys_Two(img_size=64,
                                                patch_size=4,
                                                in_chans=32,
                                                num_classes=3,
                                                embed_dim=96,
                                                depths=[2, 2, 2, 2],
                                                num_heads=[3, 6, 12, 24],
                                                window_size=8,
                                                mlp_ratio=4.,
                                                qkv_bias=True,
                                                qk_scale=None,
                                                drop_rate=0.0,
                                                drop_path_rate=0.0,
                                                ape=False,
                                                patch_norm=True,
                                                use_checkpoint=False)
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1) 

    def forward(self, x, target_time=None):
        B, T, C, H, W = x.shape
        # print(x.shape)
        # original_height, original_width = H, W
        # target_height, target_width  = 64, 64
           
        # # For Padding             
        # pad_height = target_height - original_height
        # pad_width = target_width - original_width
        # top = pad_height // 2
        # bottom = pad_height - top
        # left = pad_width // 2
        # right = pad_width - left
        
        # x = F.pad(x, (left, right, top, bottom), 'constant', 0) 
        # print("shape of x", x.shape)
        
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        # print("shape of x after rearrange",x.shape)
        residual = x
        # print("shape of residual",residual.shape)
        # print(f"Shape of x: {x.shape}")

        # print(f"Shape of residual: {residual.shape}")
    
        features= self.features(x)
        features = self.output_conv(features)
        # print("shape of tensor after  backbone",features.shape)
        out_upsampled = F.interpolate(features, size=(64, 64), mode="bilinear", align_corners=False)
        # print("shape of tensor after out_sampled")  # Keeps shape [B, 1, C, 64, 64]
        out = self.ca(x)
        # print(f"Shape of out: {out.shape}")
        # print("shape of tensor after ca",out.shape)
        x_cam = x*out + residual       
        # print("Any NaNs in x:", torch.isnan(x).any())
        # print("Any NaNs in out:", torch.isnan(out).any())
        # print("Any NaNs in residual:", torch.isnan(residual).any())
        x = rearrange(x_cam, 'b (t c) h w -> b t c h w', t=T)

        
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        # Apply FPN blocks before passing to the main model
        x_fpn1 = self.fpn1(x, x)  # FPN block 1, using x_cam as skip connection
        # print(f"x after FPN1: {x_fpn1.shape}")

        x_fpn2 = self.fpn2(x_fpn1, x_fpn1)  # FPN block 2, using x_cam as skip connection
        # print(f"x after FPN2: {x_fpn2.shape}")

        x_fpn3 = self.fpn3(x_fpn2, x_fpn2)  # FPN block 3, using x_cam as skip connection
        # print(f"x after FPN3: {x_fpn3.shape}")
             
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
            
        logits1,logits2 = self.swin_unet(x_fpn3)
        
        # For Recover Index
        # original_height, original_width = 64, 64
        # target_height, target_width  = 50, 65
                   
        # # For Padding             
        # pad_height = target_height - original_height
        # pad_width = target_width - original_width
        # top = pad_height // 2
        # bottom = pad_height - top
        # left = pad_width // 2
        # right = pad_width - left
        
        # logits1 = F.pad(logits1, (left, right, top, bottom), 'constant', 0)  
        # logits2 = F.pad(logits2, (left, right, top, bottom), 'constant', 0)
        logits1 = rearrange(logits1, '(b t) c h w -> b t c h w', c = 3, t = T).mean(dim=1)
        logits2 = rearrange(logits2, '(b t) c h w -> b t c h w', c = 1, t = T).mean(dim=1)       
        return logits1,logits2
    
    

      
# model = SwinUnet().cuda()
# # print(model)
# input = torch.randn(4, 6, 12, 50, 65).cuda()
# output = model(input)
# print(output.shape)
# 输入的是4个channel，输出的是4张图

# model = SwinUnet_CAM_Two().cuda()
# parameters = filter(lambda p: p.requires_grad, model.parameters())
# parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
# print('Trainable Parameters: %.5fM' % parameters)
# input = torch.randn(4, 1, 143, 64, 64).cuda()
# output1,output2 = model(input)
# print(output1.shape)
# print(output2.shape)
