from utils import *
from args import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

class BasicBlockWRN(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate):
        super(BasicBlockWRN, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, feature_maps, input_shape, few_shot, rotations, depth = 28, widen_factor = 10, num_classes = 64, drop_rate = args.dropout):
        super(WideResNet, self).__init__()
        nChannels = [feature_maps, feature_maps*widen_factor, 2 * feature_maps*widen_factor, 4 * feature_maps*widen_factor]
        n = (depth - 4) / 6
        self.conv1 = nn.Conv2d(input_shape[0], nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.blocks = torch.nn.ModuleList()
        self.blocks.append(NetworkBlock(n, nChannels[0], nChannels[1], BasicBlockWRN, 1, drop_rate))
        self.blocks.append(NetworkBlock(n, nChannels[1], nChannels[2], BasicBlockWRN, 2, drop_rate))
        self.blocks.append(NetworkBlock(n, nChannels[2], nChannels[3], BasicBlockWRN, 2, drop_rate))
        self.bn = nn.BatchNorm2d(nChannels[3])
        self.linear = linear(nChannels[3], int(num_classes))
        self.rotations = rotations
        self.linear_rot = linear(nChannels[3], 4)

    def forward(self, x, index_mixup = None, lam = -1):
        if lam != -1:
            mixup_layer = random.randint(0, 3)
        else:
            mixup_layer = -1
        out = x
        if mixup_layer == 0:
            out = lam * out + (1 - lam) * out[index_mixup]
        out = self.conv1(out)
        for i in range(len(self.blocks)):
            out = self.blocks[i](out)
            if mixup_layer == i + 1:
                out = lam * out + (1 - lam) * out[index_mixup]
        out = torch.relu(self.bn(out))
        out = F.avg_pool2d(out, out.size()[2:])
        out = out.view(out.size(0), -1)
        features = out
        out = self.linear(features)
        if self.rotations:
            out_rot = self.linear_rot(features)
            return (out, out_rot), features
        return out, features
