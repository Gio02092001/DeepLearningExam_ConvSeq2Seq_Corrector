import math
import torch
from torch import nn
import torch.nn.functional as F
from Model_New.LayerModules_new import LinearTransformation_New

class Attention_New(nn.Module):
    """
        Attention mechanism for the decoder.

        - Projects decoder outputs into embedding space.
        - Computes alignment scores (attention) between decoder states and encoder outputs.
        - Uses these scores to build a conditional input from encoder representations.
        - Combines conditional input with decoder state via residual connection.
    """

    def __init__(self, hidden_dim,embedding_dim, p_dropout):
        """
            Initialize attention module.

            Args:
                hidden_dim: Dimensionality of hidden states.
                embedding_dim: Dimensionality of embeddings used for attention.
                p_dropout: Dropout probability.
        """
        super().__init__()
        self.linearInput = LinearTransformation_New(False, hidden_dim, embedding_dim, p_dropout)
        self.linearOutput = LinearTransformation_New(False, embedding_dim, hidden_dim, p_dropout)

    def forward(self, decoderOutput,targetEmbedding_g,encoderOutput_z,c_inputEncoder):
        """
            Compute attention output.

            Args:
                    - decoderOutput: Tensor result from current decoder layer [batch, seq_len, hidden_dim]
                    - targetEmbedding_g: Tensor result from target embedding for reidual connection [batch, seq_len, embedding_dim]
                    - encoderOutput_z: Tensor result from last Encoder layer [batch, seq_len, hidden_dim]
                    - c_inputEncoder: Tensor result from source embedding [batch, seq_len, hidden_dim]


            Returns:
                output: Tensor resulting from attention computation [batch, decoder_len, hidden_dim]
        """

        # Project decoder output into embedding space
        decoderOutput = self.linearInput(decoderOutput)
        residual=decoderOutput

        # Combine with target embeddings to form decoder state
        decoderstate = (decoderOutput + targetEmbedding_g) * math.sqrt(0.5)

        # Compute scalar attention scores: [batch, decoder_len, encoder_len]
        try:
            scalarProducts = torch.bmm(decoderstate, encoderOutput_z.transpose(1,2))  # [batch, decoder_len, encoder_len]
        except:
            print("decoderstate:", decoderstate.shape)  # [batch, dec_len, hidden]
            print("encoderOutput_z:", encoderOutput_z.shape)  # [batch, enc_len, hidden]
        sz = scalarProducts.size()
        attentionScores = F.softmax(scalarProducts.view(sz[0] * sz[1], sz[2]), dim=1)
        attentionScores = attentionScores.view(sz)

        # Build conditional input from encoder representations
        conditionalInput =torch.bmm(attentionScores, c_inputEncoder)

        # Scale conditional input by sequence length
        s = c_inputEncoder.size(1)
        conditionalInput = conditionalInput * math.sqrt(s)

        # Residual connection
        output = (residual+conditionalInput) * math.sqrt(0.5)

        # Project back to hidden dimension
        return self.linearOutput(output)
