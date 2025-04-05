import math
import torch
from torch import nn
import torch.nn.functional as F

from Model_New.Attention_New import Attention_New
from Model_New.LayerModules_new import ConvolutionalBlock_New


class Decoder_Attention_New(nn.Module):
    def __init__(self, hidden_dim,embedding_dim, p_dropout, kernel_width):
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.conv = ConvolutionalBlock_New(hidden_dim, kernel_width, p_dropout)
        self.kernel_width = kernel_width

        self.attention = Attention_New (hidden_dim,embedding_dim, p_dropout)

    def forward(self, input):
        inputDecoder = input[0]
        targetEmbedding_g = input[1]
        encoderOutput_z = input[2]
        c_inputEncoder = input[3]
        residual=inputDecoder
        seq_Length = inputDecoder.size(1)

        pad = self.kernel_width - 1
        inputDecoder = inputDecoder.transpose(1, 2)
        inputDecoder = F.pad(inputDecoder, pad=(pad, pad), mode='constant', value=0)
        inputDecoder=self.conv(inputDecoder)
        inputDecoder=inputDecoder.transpose(1, 2)
        inputDecoder = inputDecoder[:, :seq_Length, :]

        outputAttention = self.attention(inputDecoder,targetEmbedding_g,encoderOutput_z,c_inputEncoder)

        decoder_attentionOutput= (outputAttention + residual) * math.sqrt(0.5)

        return (decoder_attentionOutput, targetEmbedding_g, encoderOutput_z, c_inputEncoder)



"""
class Decoder(nn.Module):
    def __init__(self, hidden_dim, p_dropout, kernel_width):
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.conv = ConvolutionalBlock(hidden_dim, kernel_width, p_dropout)
        self.kernel_width = kernel_width
        self.dropOutput = nn.Dropout(p_dropout)

    def forward(self, input):
        residual = input
        pad = self.kernel_width - 1

        # Applicazione del dropout
        input = self.drop(input)

        # Transposizione per allineare le dimensioni
        input = input.transpose(0, 1)

        # Padding dell'input
        input = F.pad(input, pad=(int(pad), int(pad)), mode='constant', value=0)

        tensor = []
        input = input.transpose(0, 1)

        # Applicazione della convoluzione per ciascun step
        for i in range(input.shape[0] - pad * 2):
            conv_input = input[i:i + pad + 1]
            conv_input = conv_input.transpose(0, 1)
            tensor.append(self.conv(conv_input))

        # Combinazione dei risultati e restituzione dell'output
        result = (residual + torch.stack(tensor)) * math.sqrt(0.5)
        return self.dropOutput(result)


    def forward(self, input):
        # input shape: [batch_size, seq_len, features]
        batch_size, seq_len, features = input.size()
        residual = input
        pad = self.kernel_width - 1

        # Apply dropout to input
        input = self.drop(input)

        # Transpose and pad for convolution
        # From [batch_size, seq_len, features] to [batch_size, features, seq_len]
        input = input.transpose(1, 2)

        # Apply padding to sequence dimension for all batches at once
        # This adds padding to both sides of the sequence dimension
        input = F.pad(input, pad=(int(pad), int(pad)), mode='constant', value=0)
        # Now input shape is [batch_size, features, seq_len + 2*pad]

        # Transpose back to [batch_size, seq_len + 2*pad, features]
        input = input.transpose(1, 2)

        # Process each position in the sequence for all batches at once
        tensor = []
        for i in range(seq_len):  # Loop only through original sequence length
            # Extract window for all batches at this position
            # Window shape: [batch_size, kernel_width, features]
            conv_window = input[:, i:i + pad + 1, :]

            # Transpose to prepare for convolution
            # From [batch_size, kernel_width, features] to [batch_size, features, kernel_width]
            conv_window = conv_window.transpose(1, 2)

            # Apply convolution to all batches at once
            # Output shape: [batch_size, output_features, 1]
            result = self.conv(conv_window)

            # Squeeze the last dimension and add to results
            # Result shape: [batch_size, output_features]
            tensor.append(result.squeeze(-1))

        # Stack the results along sequence dimension
        # Shape: [batch_size, seq_len, output_features]
        stacked_output = torch.stack(tensor, dim=1)

        # Combine with residual connection
        result = (residual + stacked_output) * math.sqrt(0.5)

        # Apply output dropout and return
        return self.dropOutput(result)"""