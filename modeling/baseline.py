# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.efficientnet import EfficientNet
from .backbones.stn import STN
import sys
from .models_lpf import resnet as resnet_lpf


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)



def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1,
                 act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class neck(nn.Module):
    def __init__(self,c1,nc,dropout= 0.5):
        super().__init__()
        self.conv1 = Conv(c1,c1)
        self.conv2 = Conv(c1,c1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=dropout, inplace=True)
        self.line = nn.Linear(c1, nc)

    def forward(self,x):

        x = self.drop(self.gap(self.conv2(self.conv1(x))))
        global_feat = x.view(x.shape[0], -1)  # flatten to (bs, 2048)
        # global_feat = x.view(1, -1)  # flatten to (bs, 2048)   # to_onnx   bs=1

        return self.line(global_feat)





class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, last_stride, model_path, model_name, pretrain_choice, cfg):
        super(Baseline, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 6, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 23, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)  
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=64, 
                              reduction=16,
                              dropout_p=0.2, 
                              last_stride=last_stride)
        elif 'efficientnet' in model_name:
            self.base = EfficientNet.from_name(model_name)
            self.in_planes = self.base.in_planes
        elif model_name == 'resnet50_lpf':
            #import pdb;pdb.set_trace()
            self.base = resnet_lpf.resnet50(filter_size=5)


        #import pdb;pdb.set_trace()
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model, model name {}, model path {}......'.format(model_name, model_path))


        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)

        self.with_stn = cfg.MODEL.IF_WITH_STN
        self.stn = STN()


        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        #import pdb;pdb.set_trace()


        self.country_cls = neck(self.in_planes, cfg.MODEL.nc1,dropout=cfg.MODEL.DROPOUT)        # 国家分类
        self.cn_cls      = neck(self.in_planes, cfg.MODEL.nc2,dropout=cfg.MODEL.DROPOUT)        # 号码区域
        self.ct_cls      = neck(self.in_planes, cfg.MODEL.nc3,dropout=cfg.MODEL.DROPOUT)        # 边条颜色

        if pretrain_choice == 'frozen':
            for p in self.parameters():
                p.requires_grad=False


        #self.brand_cls.apply(weights_init_classifier)
        if pretrain_choice == 'frozen':
            print('Loading pretrained model, model name {}, model path {}......'.format(model_name, model_path))
            self.load_param(model_path)

        if pretrain_choice == 'finetune':
            self.load_param(model_path)
            print('Loading pretrained model to finetune, model name {}, model path {}......'.format(model_name, model_path))


    def forward(self, x):

        if self.with_stn == 'yes':
            x = self.stn(x)

        global_feat = self.base(x)  # (b, 2048, 1, 1)

        country_score = self.country_cls(global_feat)
        cn_score = self.cn_cls(global_feat)
        ct_score = self.ct_cls(global_feat)

        return (country_score, cn_score, ct_score)


    def load_param(self, trained_path, device='cuda:0'):
        param_dict = torch.load(trained_path, map_location=device)
        for i in param_dict:
            #if 'classifier' in i:
            if i not in self.state_dict():
                print('not load param ', i)
                continue
            self.state_dict()[i].copy_(param_dict[i])
