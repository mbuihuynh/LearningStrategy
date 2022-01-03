import torch
import torch.nn.functional as F
import torch.nn as nn


'''
    Explained: https://towardsdatascience.com/advances-in-few-shot-learning-reproducing-results-in-pytorch-aba70dee541d
    Study the coding from github: https://github.com/oscarknagg/few-shot

'''


def conv_block(in_channels:int, out_channels:int ) -> nn.Module:

    block = nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    return block

class ClassifierBase(nn.Module):
    def __init__(self, num_in_channels:int , output_class:int, ):
        super(ClassifierBase).__init__()

        self.conv1 = conv_block(num_in_channels,64)
        self.conv2 = conv_block(64,64)
        self.conv3 = conv_block(64,64)
        self.conv4 = conv_block(64,64)
        self.logits = nn.Linear(64,output_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv1(x)
        x = x.view(x.size(0),-1)
        out = self.logits(x)

        return out

