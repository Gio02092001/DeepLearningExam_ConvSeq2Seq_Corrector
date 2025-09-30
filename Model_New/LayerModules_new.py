import math
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

class LinearTransformation_New(nn.Module):
    """
       Linear transformation module with optional dropout-aware initialization
       and weight normalization.
    """
    def __init__(self, noDropoutBefore, input_dim, output_dim, p_dropout):
        """
            Initialize the linear transformation.

            Args:
                no_dropout_before (bool): Whether to ignore dropout in weight initialization.
                input_dim (int): Size of input features.
                output_dim (int): Size of output features.
                p_dropout (float): Dropout probability for modified initialization.
        """
        super().__init__()

        # Define a weight-normalized linear layer
        self.linear = weight_norm(
            nn.Linear(input_dim, output_dim),
            name='weight', dim=0
        )

        # Initialize weights with consideration of dropout to counter variance change
        if noDropoutBefore:
            init.normal_(self.linear.weight, mean=0.0, std=math.sqrt(1 / input_dim))
        else:
            init.normal_(self.linear.weight, mean=0.0, std=math.sqrt((1-p_dropout) / input_dim))
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, input):
        return self.linear(input)

class ConvolutionalBlock_New(nn.Module):
    """
        1D Convolutional block with GLU activation and weight normalization.

        This block doubles the hidden dimension, applies GLU along channel dimension,
        and is initialized based on kernel width, hidden_dim, and dropout.
    """
    def __init__(self, hidden_dim, kernel_width, p_dropout):
        """
            Initialize the convolutional block.

            Args:
                hidden_dim (int): Number of input channels.
                kernel_width (int): Width of the convolutional kernel.
                p_dropout (float): Dropout probability for weight initialization scaling.
        """
        super().__init__()

        # Weight initialization considering GLU doubling and dropout
        self.conv = weight_norm(
            nn.Conv1d(in_channels=hidden_dim,
                              out_channels=hidden_dim * 2,
                              kernel_size=kernel_width,
                              padding=0,
                              stride=1),
            name='weight', dim=0
        )
        torch.nn.init.normal_(self.conv.weight, mean=0.0,
                              std=math.sqrt(4 * (1-p_dropout) / (hidden_dim * kernel_width)))
        nn.init.constant_(self.conv.bias, 0)


    def forward(self, input):
        """
            Forward pass through convolution and GLU activation.

            Args:
                input (Tensor): Input tensor of shape [batch, hidden_dim, seq_len].

            Returns:
                Tensor: Output tensor after GLU activation [batch, hidden_dim, seq_len].
        """
        input = self.conv(input)
        return F.glu(input, 1)
