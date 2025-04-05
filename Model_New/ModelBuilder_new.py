import math

from torch import nn

from Model_New.Classification_New import Classification_New
from Model_New.Decoder_New import Decoder_Attention_New
from Model_New.Encoder_New import Encoder_New
from Model_New.InitialEmbedding_new import InitialEmbedding_New
from Model_New.LayerModules_new import LinearTransformation_New


class ConvModel_New(nn.Module):
    def __init__(self, target_vocab_size, vocab_size, fixedNumberOfInputElements, embedding_dim, p_dropout, hidden_dim,
                 kernel_width, encoderLayer, decoderLayer, unknown_token, device):
        super().__init__()
        # Encoder and its embedding
        self.device=device
        self.embeddingEncoder = InitialEmbedding_New(vocab_size, fixedNumberOfInputElements, embedding_dim,
                                                     p_dropout, hidden_dim, unknown_token)

        self.linearEncoderInput = LinearTransformation_New(False, embedding_dim, hidden_dim, p_dropout)
        self.encoderBlocks = nn.Sequential(
            *[(Encoder_New(embedding_dim, hidden_dim, p_dropout, kernel_width)) for _ in range(encoderLayer)]
        )
        self.linearEncoderOutput = LinearTransformation_New(True, hidden_dim, embedding_dim, p_dropout)


        self.embeddingDecoder = InitialEmbedding_New(target_vocab_size, fixedNumberOfInputElements, embedding_dim,
                                                 p_dropout, hidden_dim, unknown_token)


        self.linearDecoderInput = LinearTransformation_New(False, embedding_dim, hidden_dim, p_dropout)
        self.decoder_attentionBlocks = nn.Sequential(
            *[(Decoder_Attention_New(hidden_dim, embedding_dim, p_dropout, kernel_width)) for _ in range(decoderLayer)]
        )
        self.classification = Classification_New(hidden_dim, embedding_dim, target_vocab_size, p_dropout)
        self.attentionLayer=decoderLayer
    def forward(self, source_input, target_input):
        inputEmbedding_w = self.embeddingEncoder(source_input)
        inputEncoder = self.linearEncoderInput(inputEmbedding_w)
        encoderOutput_z=self.encoderBlocks(inputEncoder)
        encoderOutput_z=self.linearEncoderOutput(encoderOutput_z)

        scale = 1.0 / (2.0 * self.attentionLayer)
        encoderOutput_z = encoderOutput_z * scale + encoderOutput_z.detach() * (1 - scale)

        c_inputEncoder= (inputEmbedding_w+encoderOutput_z)*math.sqrt(0.5)


        targetEmbedding_g = self.embeddingDecoder(target_input)

        inputDecoder = self.linearDecoderInput(inputEmbedding_w)

        decoder_attentionOutput = self.decoder_attentionBlocks(
            (inputDecoder, targetEmbedding_g, encoderOutput_z, c_inputEncoder)
        )
        output=self.classification(decoder_attentionOutput)
        return output




