import math
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import model_urls
import torchvision.models as models
from torch.nn import functional as F
from .base import Backbone

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)

# class Inter_Atten(nn.Module):
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, use_bn=True):
#         super(Inter_Atten, self).__init__()
#
#     def forward(self, x):

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class Cha_Spa(nn.Module):
    def __init__(self):
        super(Cha_Spa, self).__init__()

        self.cha1 = ChannelAttention(128)
        self.cha2 = ChannelAttention(256)

        self.spa1 = SpatialAttention()
        self.spa2 = SpatialAttention()

    def forward(self, frame, event):
        # print('ef', xf[0].size(), ef[0].size())

        temp1 = event[0].mul(self.cha1(event[0]))
        temp1 = temp1.mul(self.spa1(temp1))
        x1 = frame[0] + temp1
        temp2 = event[1].mul(self.cha2(event[1]))
        temp2 = temp2.mul(self.spa2(temp2))
        x2 = frame[1] + temp2
        return [x1, x2]