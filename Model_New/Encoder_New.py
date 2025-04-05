import math
import torch
import torch.nn.functional as F

from torch import nn

from Model_New.LayerModules_new import LinearTransformation_New, ConvolutionalBlock_New


class Encoder_New(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, p_dropout, kernel_width):
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.conv = ConvolutionalBlock_New(hidden_dim, kernel_width, p_dropout)
        self.kernel_width = kernel_width

    def forward(self, input):
        input=self.drop(input)
        input=input.transpose(1, 2)
        residual = input
        pad=int((self.kernel_width-1)/2)
        input = F.pad(input, pad=(pad,pad), mode='constant', value=0)
        input = self.conv(input)
        input = (residual + input) * math.sqrt(0.5)
        input=input.transpose(1,2)
        return input




