import torch
import torch.nn as nn


class MaxConv2d(nn.Module):
    """Basic convolutional block.
    
    convolution + max + relu.
    Args:
        in_c (int): number of input channels.
        out_c (int): number of output channels.
        k (int or tuple): kernel size.
        s (int or tuple): stride.
        p (int or tuple): padding.
    """
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(MaxConv2d, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        nn.init.xavier_normal_(self.conv.weight)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.bn(self.conv(x))
        x, _ = torch.max(x, dim=1)
        x, _ = torch.max(x, dim=1)
        x = torch.relu(x)
        return x


class AvgConv2d(nn.Module):
    """Basic convolutional block.
    
    convolution + avg + relu.
    Args:
        in_c (int): number of input channels.
        out_c (int): number of output channels.
        k (int or tuple): kernel size.
        s (int or tuple): stride.
        p (int or tuple): padding.
    """
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(AvgConv2d, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        nn.init.xavier_normal_(self.conv.weight)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = torch.mean(x, dim=1)
        x = torch.mean(x, dim=1)
        x = torch.relu(x)
        return x


class SigmoidConv2d(nn.Module):
    """Basic convolutional block.
    
    convolution + sigmoid
    Args:
        in_c (int): number of input channels.
        out_c (int): number of output channels.
        k (int or tuple): kernel size.
        s (int or tuple): stride.
        p (int or tuple): padding.
    """
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(SigmoidConv2d, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        nn.init.xavier_normal_(self.conv.weight)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = torch.sigmoid(x)
        return x
