import math

import torch
from mpmath import residual
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from Model_New.LayerModules_new import LinearTransformation_New


class Attention_New(nn.Module):
    def __init__(self, hidden_dim,embedding_dim, p_dropout):
        super().__init__()
        self.linearInput = LinearTransformation_New(False, hidden_dim, embedding_dim, p_dropout)
        self.linearOutput = LinearTransformation_New(False, embedding_dim, hidden_dim, p_dropout)

    def forward(self, decoderOutput,targetEmbedding_g,encoderOutput_z,c_inputEncoder):

        decoderOutput = self.linearInput(decoderOutput)
        residual=decoderOutput

        decoderstate = decoderOutput + targetEmbedding_g   # )*math. #sqrt(0.5)  # Decoder state with residual connection

        scalarProducts = torch.bmm(decoderstate, encoderOutput_z.transpose(1,2))  # [batch, decoder_len, encoder_len]
        sz = scalarProducts.size()
        attentionScores = F.softmax(scalarProducts.view(sz[0] * sz[1], sz[2]), dim=1)
        attentionScores = attentionScores.view(sz)

        conditionalInput =torch.bmm(attentionScores, c_inputEncoder)
        s = c_inputEncoder.size(1)
        conditionalInput = conditionalInput * s  #* math.sqrt(1.0 / s))


        output = residual+conditionalInput #)*math.sqrt(0.5)

        return self.linearOutput(output)
