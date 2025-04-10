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
from torchvision import models



class SwinUnet(nn.Module):
    def _init_(self, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet, self)._init_()
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
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        logits = self.swin_unet(x)
        return logits
    
  
class SwinUnet_Two(nn.Module):
    def _init_(self, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet_Two, self)._init_()
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
        logits1 = rearrange(logits1, '(b t) c h w -> b t c h w', c = 3, t = 6).mean(dim=1)
        logits2 = rearrange(logits2, '(b t) c h w -> b t c h w', c = 1, t = 6).mean(dim=1)       
        return logits1,logits2

   
class SwinUnet_CAM_Two(nn.Module):
    def _init_(self, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet_CAM_Two, self)._init_()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.ca = ChannelAttention(in_planes=1*143, ratio=16)         
        self.swin_unet = SwinTransformerSys_Two(img_size=64,
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
        # print(x.shape)
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
        
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        residual = x
        out = self.ca(x)
        x_cam = x*out + residual       
        x = rearrange(x_cam, 'b (t c) h w -> b t c h w', t=T)
        
        x = rearrange(x, 'b t c h w -> (b t) c h w')
             
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
            
        logits1,logits2 = self.swin_unet(x)
        
        # # For Recover Index
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

class SwinUnet_Two(nn.Module):
    def _init_(self, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet_Two, self)._init_()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.swin_unet = SwinTransformerSys_Two(img_size=64,
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

class FPNBlock(nn.Module):
    def _init_(self, in_channels, out_channels):
        super(FPNBlock, self)._init_()
        
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
        # First, upsample the input tensor x
        x = self.upsample(x)
        # print(f"Shape of x: {x.shape}")
        # print(f"Shape of skip: {skip.shape}")

        
        # Check if the spatial size of x and skip match, if not, adjust it
        if x.shape[2:] != skip.shape[2:]:
            skip = nn.functional.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        # Ensure skip connection has the same channel depth as the output tensor (through 1x1 convolution)
        skip = self.skip_conv(skip)  # Apply 1x1 convolution to match the channels
        
        # Now, add the x and skip tensors (they should have the same channel size)
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
from torchvision import models
from einops import rearrange

class SwinUnet_CAM_Two(nn.Module):
    def __init__(self, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet_CAM_Two, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        
        self.ca = CBAMBlock(channel=12 * 6, reduction=16)

        self.efficientnet = models.efficientnet_b4(weights="IMAGENET1K_V1")
        self.efficientnet.features[0][0] = nn.Conv2d(72, 48, kernel_size=3, stride=2, padding=1, bias=False)
        last_channel = self.efficientnet.features[-1][0].out_channels

        self.output_conv = nn.Conv2d(last_channel, 72, kernel_size=1, stride=1, padding=0, bias=False)

        self.features = self.efficientnet.features    

        # self.fpn1 = FPNBlock(12, 96)  # Change input from 72 → 12
        # self.fpn2 = FPNBlock(96, 64)
        # self.fpn3 = FPNBlock(64, 32)
  

        self.swin_unet = SwinTransformerSys_Two(
            img_size=64,
            patch_size=4,
            in_chans=12,
            num_classes=3,
            embed_dim=96,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            window_size=4,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            drop_path_rate=0.0,
            ape=False,
            patch_norm=True,
            use_checkpoint=False
        )

    def forward(self, x, target_time=None):
        B, T, C, H, W = x.shape
        target_height, target_width = 64, 64

        # Padding input to target shape
        pad_height = target_height - H
        pad_width = target_width - W
        top, bottom = pad_height // 2, pad_height - (pad_height // 2)
        left, right = pad_width // 2, pad_width - (pad_width // 2)

        x = F.pad(x, (left, right, top, bottom), "constant", 0)
        print("come in model class")
        
        # Reshape for EfficientNet
        x = rearrange(x, "b t c h w -> b (t c) h w")
        residual = x
        features = self.features(x)

        features = self.output_conv(features)
        out_upsampled = F.interpolate(features, size=(224, 224), mode="bilinear", align_corners=False)

        out = self.ca(x)
        x_cam = x * out + residual
        x = rearrange(x_cam, "b (t c) h w -> b t c h w", t=T)

        x = rearrange(x, "b t c h w -> (b t) c h w")
       

        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        logits1, logits2 = self.swin_unet(x)

        # Recover Index
        target_height, target_width = 50, 65
        pad_height = target_height - 64
        pad_width = target_width - 64
        top, bottom = pad_height // 2, pad_height - (pad_height // 2)
        left, right = pad_width // 2, pad_width - (pad_width // 2)

        logits1 = F.pad(logits1, (left, right, top, bottom), "constant", 0)
        logits2 = F.pad(logits2, (left, right, top, bottom), "constant", 0)

        logits1 = rearrange(logits1, "(b t) c h w -> b t c h w", c=3, t=T).mean(dim=1)
        logits2 = rearrange(logits2, "(b t) c h w -> b t c h w", c=1, t=T).mean(dim=1)

        return logits1, logits2



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import models
# from einops import rearrange

# # Define ConvLSTM module
# class ConvLSTM(nn.Module):
#     def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=True):
#         super(ConvLSTM, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers

#         # Corrected: Using Conv2d LSTM (ConvLSTMCell for each layer)
#         self.conv_lstm = nn.LSTM(input_size=12 * 64 * 64, hidden_size=hidden_dim, num_layers=num_layers, batch_first=batch_first)
        
#     def forward(self, x):
#         # x shape: (B, T, C, H, W)
#         B, T, C, H, W = x.size()

#         # Flatten spatial dimensions but preserve channel and time
#         x = x.view(B, T, C * H * W)  # Shape: (B, T, 12 * 64 * 64 = 49152)

#         # Forward pass through ConvLSTM
#         output, _ = self.conv_lstm(x)

#         # Take output of the last time step
#         output = output[:, -1, :]  # Shape: (B, hidden_dim)

#         # Reshape back to (B, C, H, W)
#         output = output.view(B, self.hidden_dim, 1, 1)
#         return output

# # SwinUnet_CAM_Two class corrected
# class SwinUnet_CAM_Two(nn.Module):
#     def __init__(self, img_size=224, num_classes=21843, zero_head=False, vis=False):
#         super(SwinUnet_CAM_Two, self).__init__()
#         self.num_classes = num_classes
#         self.zero_head = zero_head

#         # CBAMBlock (unchanged)
#         self.ca = CBAMBlock(channel=12 * 6, reduction=16)

#         # Corrected ConvLSTM Backbone
#         self.convlstm = ConvLSTM(input_dim=12, hidden_dim=48, kernel_size=(3, 3), num_layers=2)

#         # Output layer
#         self.output_conv = nn.Conv2d(48, 72, kernel_size=1, stride=1, padding=0, bias=False)
#         # Swin Transformer
#         self.swin_unet = SwinTransformerSys_Two(
#             img_size=64,
#             patch_size=4,
#             in_chans=12,
#             num_classes=3,
#             embed_dim=96,
#             depths=[2, 2, 2, 2],
#             num_heads=[3, 6, 12, 24],
#             window_size=4,
#             mlp_ratio=4.,
#             qkv_bias=True,
#             qk_scale=None,
#             drop_rate=0.0,
#             drop_path_rate=0.0,
#             ape=False,
#             patch_norm=True,
#             use_checkpoint=False
#         )

#     def forward(self, x, target_time=None):
#         B, T, C, H, W = x.shape
#         target_height, target_width = 64, 64

#         # print("shape of input", x.shape)

#         # Padding input to target shape
#         pad_height = target_height - H
#         pad_width = target_width - W
#         top, bottom = pad_height // 2, pad_height - (pad_height // 2)
#         left, right = pad_width // 2, pad_width - (pad_width // 2)

#         x = F.pad(x, (left, right, top, bottom), "constant", 0)

#         # print("shape before lstm", x.shape)

#         # Pass through ConvLSTM backbone
#         features = self.convlstm(x)
#         # print("shape after lstm immediately", features.shape)

#         features = self.output_conv(features)
#         # print("shape after output conv", features.shape)
#         out_upsampled = F.interpolate(features, size=(224, 224), mode="bilinear", align_corners=False)

#         # Channel Attention
#         out = self.ca(features)
#         x_cam = features * out
#         x = rearrange(x_cam, "b (t c) h w -> b t c h w", t=T)

#         x = rearrange(x, "b t c h w -> (b t) c h w")

#         x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)

#         if x.size(1) == 1:
#             x = x.repeat(1, 3, 1, 1)

#         logits1, logits2 = self.swin_unet(x)

#         # Recover Index
#         target_height, target_width = 50, 65
#         pad_height = target_height - 64
#         pad_width = target_width - 64
#         top, bottom = pad_height // 2, pad_height - (pad_height // 2)
#         left, right = pad_width // 2, pad_width - (pad_width // 2)

#         logits1 = F.pad(logits1, (left, right, top, bottom), "constant", 0)
#         logits2 = F.pad(logits2, (left, right, top, bottom), "constant", 0)

#         logits1 = rearrange(logits1, "(b t) c h w -> b t c h w", c=3, t=T).mean(dim=1)
#         logits2 = rearrange(logits2, "(b t) c h w -> b t c h w", c=1, t=T).mean(dim=1)

#         return logits1, logits2




# class SwinUnet_CAM_Two(nn.Module):
#     def _init_(self, img_size=224, num_classes=21843, zero_head=False, vis=False):
#         super(SwinUnet_CAM_Two, self)._init_()
#         self.num_classes = num_classes
#         self.zero_head = zero_head
#         self.ca = CBAMBlock(channel=72, reduction=16)       
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
#         # print(x.shape)
#         original_height, original_width = H, W
#         target_height, target_width  = 64, 64
           
#         # # For Padding             
#         pad_height = target_height - original_height
#         pad_width = target_width - original_width
#         top = pad_height // 2
#         bottom = pad_height - top
#         left = pad_width // 2
#         right = pad_width - left
        
#         x = F.pad(x, (left, right, top, bottom), 'constant', 0) 
        
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
#         logits1 = rearrange(logits1, '(b t) c h w -> b t c h w', c = 3, t = T).mean(dim=1)
#         logits2 = rearrange(logits2, '(b t) c h w -> b t c h w', c = 1, t = T).mean(dim=1)       
#         return logits1,logits2

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.models.video import r3d_18  # Using ResNet3D-18, can be replaced with deeper variants

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from torchvision.models.video import r3d_18, R3D_18_Weights # or any other ResNet3D variant

# # Assuming you have CBAMBlock defined





import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AdaptiveAvgPool3d
from torchvision.models.video import r3d_18, R3D_18_Weights
from einops import rearrange

# Define your SwinTransformerSys_Two here (assuming it is already implemented)
# from your_module import SwinTransformerSys_Two



import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import regnet_y_400mf  # Replace with other variants if needed
from einops import rearrange











    
    
      
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