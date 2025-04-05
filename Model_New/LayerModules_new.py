import math
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F


class LinearTransformation_New(nn.Module):
    def __init__(self, noDropoutBefore, input_dim, output_dim, p_dropout):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        if noDropoutBefore:
            init.normal_(self.linear.weight, mean=0.0, std=math.sqrt(1 / input_dim))  # He initialization
        else:
            init.normal_(self.linear.weight, mean=0.0, std=math.sqrt((1-p_dropout) / input_dim))  # Modified initialization
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, input):
        return self.linear(input)


class ConvolutionalBlock_New(nn.Module):
    def __init__(self, hidden_dim, kernel_width, p_dropout):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=hidden_dim,
                              out_channels=hidden_dim * 2,
                              kernel_size=kernel_width,
                              padding=0,  # Adding padding to maintain the sequence length
                              stride=1)
        torch.nn.init.normal_(self.conv.weight, mean=0.0,
                              std=math.sqrt(4 * (1-p_dropout) / (hidden_dim * kernel_width)))
        nn.init.constant_(self.conv.bias, 0)


    def forward(self, input):
        input = self.conv(input)
        return F.glu(input, 1)
