import math
import torch.nn.functional as F
from torch import nn
from Model_New.LayerModules_new import ConvolutionalBlock_New

class Encoder_New(nn.Module):
    """
        Convolutional encoder block with residual connections.

        This module:
          1. Applies dropout to the input embeddings.
          2. Performs a 1D convolution (via ConvolutionalBlock_New).
          3. Adds a residual connection with input normalization (scaled by sqrt(0.5)).
          4. Preserves the [batch, seq_len, hidden_dim] shape for downstream processing.
    """

    def __init__(self, hidden_dim, p_dropout, kernel_width):
        """
            Args:
                hidden_dim: Dimensionality of hidden representations.
                p_dropout: Dropout probability.
                kernel_width: Width of the convolutional kernel.
        """
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.conv = ConvolutionalBlock_New(hidden_dim, kernel_width, p_dropout)
        self.kernel_width = kernel_width

    def forward(self, input):
        """
            Args:
                input: Tensor of shape [batch_size, seq_len, hidden_dim]

            Returns:
                input: Tensor of shape [batch_size, seq_len, hidden_dim]
        """
        # Apply dropout
        input=self.drop(input)

        # Rearrange for convolution: [batch, hidden_dim, seq_len]
        input=input.transpose(1, 2)
        residual = input

        # Pad to preserve sequence length after convolution
        pad=int((self.kernel_width-1)/2)
        input = F.pad(input, pad=(pad,pad), mode='constant', value=0)

        # Apply convolution
        input = self.conv(input)

        # Residual connection + normalization
        input = (residual + input) * math.sqrt(0.5)

        # Transpose back to [batch, seq_len, hidden_dim]
        input=input.transpose(1,2)

        return input