import math
from torch import nn
import torch.nn.functional as F
from Model_New.Attention_New import Attention_New
from Model_New.LayerModules_new import ConvolutionalBlock_New

class Decoder_Attention_New(nn.Module):
    """
        Decoder block that combines convolution and attention.

        - Uses convolution to capture local dependencies in the decoder sequence.
        - Uses attention to align decoder states with encoder outputs.
        - Applies residual connections with normalization for stability.
    """
    def __init__(self, hidden_dim,embedding_dim, p_dropout, kernel_width):
        """
            Initialize the decoder with convolution and attention.

            Args:
                hidden_dim: Dimensionality of hidden states.
                embedding_dim: Dimensionality of target embeddings.
                p_dropout: Dropout probability.
                kernel_width: Convolution kernel size.
        """
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.conv = ConvolutionalBlock_New(hidden_dim, kernel_width, p_dropout)
        self.kernel_width = kernel_width
        self.attention = Attention_New (hidden_dim,embedding_dim, p_dropout)

    def forward(self, input):
        """
            Forward pass through the decoder.

            Args:
                input: Tuple containing
                    - inputDecoder: Tensor result from target embedding [batch, seq_len, hidden_dim]
                    - targetEmbedding_g: Tensor result from target embedding for reidual connection [batch, seq_len, embedding_dim]
                    - encoderOutput_z: Tensor result from last Encoder layer [batch, seq_len, hidden_dim]
                    - c_inputEncoder: Tensor result from source embedding [batch, seq_len, hidden_dim]

            Returns:
                Tuple containing
                    - decoder_attentionOutput: Tensor result of decoder layer [batch, seq_len, hidden_dim]
                    - targetEmbedding_g: Tensor result from target embedding for reidual connection [batch, seq_len, embedding_dim]
                    - encoderOutput_z: Tensor result from last Encoder layer [batch, seq_len, hidden_dim]
                    - c_inputEncoder: Tensor result from source embedding [batch, seq_len, hidden_dim]

        """
        inputDecoder = input[0]
        targetEmbedding_g = input[1]
        encoderOutput_z = input[2]
        c_inputEncoder = input[3]

        # Dropout on decoder input
        inputDecoder = self.drop(inputDecoder)
        residual=inputDecoder
        seq_Length = inputDecoder.size(1)

        # Convolution with padding (preserve sequence length)
        pad = self.kernel_width - 1
        inputDecoder = inputDecoder.transpose(1, 2)  # [batch, hidden_dim, seq_len]
        inputDecoder = F.pad(inputDecoder, pad=(pad, pad), mode='constant', value=0)
        inputDecoder=self.conv(inputDecoder)
        inputDecoder=inputDecoder.transpose(1, 2) # [batch, seq_len, hidden_dim]
        inputDecoder = inputDecoder[:, :seq_Length, :] # Cut padding due to avoid seeing future tokens

        # Attention mechanism over encoder outputs
        outputAttention = self.attention(inputDecoder,targetEmbedding_g,encoderOutput_z,c_inputEncoder)

        # Residual connection + normalization
        decoder_attentionOutput= (outputAttention + residual) * math.sqrt(0.5)

        return (decoder_attentionOutput, targetEmbedding_g, encoderOutput_z, c_inputEncoder)