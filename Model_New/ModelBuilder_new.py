import math
from torch import nn
from Model_New.Classification_New import Classification_New
from Model_New.Decoder_New import Decoder_Attention_New
from Model_New.Encoder_New import Encoder_New
from Model_New.InitialEmbedding_new import InitialEmbedding_New
from Model_New.LayerModules_new import LinearTransformation_New

class ConvModel_New(nn.Module):
    """
    Convolutional Sequence-to-Sequence Model with Attention.

    Based on Gehring et al., 2017, this model combines:
        - Initial embeddings for encoder and decoder
        - Convolutional blocks in the encoder and decoder
        - Attention mechanism in the decoder
        - Final classification layer to predict target vocabulary
    """

    def __init__(self, target_vocab_size, vocab_size, fixedNumberOfInputElements, embedding_dim, p_dropout, hidden_dim,
                 kernel_width, encoderLayer, decoderLayer, unknown_token, device):
        """
            Initialize the convolutional sequence-to-sequence model.

            Args:
                target_vocab_size (int): Size of the target vocabulary.
                vocab_size (int): Size of the source vocabulary.
                fixedNumberOfInputElements (int): Maximum sequence length.
                embedding_dim (int): Dimension of the embeddings.
                p_dropout (float): Dropout probability.
                hidden_dim (int): Hidden dimension for convolutional blocks.
                kernel_width (int): Width of convolutional kernels.
                encoderLayer (int): Number of encoder convolutional blocks.
                decoderLayer (int): Number of decoder convolutional blocks with attention.
                unknown_token (int): Token index for unknown words.
                device (torch.device): Device (CPU/GPU) to run the model on.
        """
        super().__init__()

        # --- Encoder ---
        # Initial embedding for encoder input sequence
        self.device=device
        self.embeddingEncoder = InitialEmbedding_New(vocab_size, fixedNumberOfInputElements, embedding_dim,
                                                     p_dropout, hidden_dim, unknown_token)

        # Linear projection of embeddings into hidden_dim for convolutional blocks
        self.linearEncoderInput = LinearTransformation_New(False, embedding_dim, hidden_dim, p_dropout)

        # Encoder convolutional blocks (sequence of Encoder_New modules)
        self.encoderBlocks = nn.Sequential(
            *[(Encoder_New( hidden_dim, p_dropout, kernel_width)) for _ in range(encoderLayer)]
        )

        # Linear projection of encoder output back to embedding_dim
        self.linearEncoderOutput = LinearTransformation_New(True, hidden_dim, embedding_dim, p_dropout)

        # --- Decoder ---
        # Initial embedding for target sequence (decoder input)
        self.embeddingDecoder = InitialEmbedding_New(target_vocab_size, fixedNumberOfInputElements, embedding_dim,
                                                 p_dropout, hidden_dim, unknown_token)

        # Linear projection of decoder embeddings into hidden_dim
        self.linearDecoderInput = LinearTransformation_New(False, embedding_dim, hidden_dim, p_dropout)

        # Decoder convolutional blocks with attention
        self.decoder_attentionBlocks = nn.Sequential(
            *[(Decoder_Attention_New(hidden_dim, embedding_dim, p_dropout, kernel_width)) for _ in range(decoderLayer)]
        )

        # Classification layer to project hidden_dim to target_vocab_size
        self.classification = Classification_New(hidden_dim, embedding_dim, target_vocab_size, p_dropout)

        # Number of attention layers (used for residual scaling)
        self.attentionLayer=decoderLayer

    def forward(self, source_input, target_input):
        """
            Forward pass of the convolutional seq2seq model.

            Args:
                source_input (Tensor): Source sequence indices [batch, seq_len].
                target_input (Tensor): Target sequence indices [batch, seq_len].

            Returns:
                Tuple: (prediction, logits) from the final classification layer.
        """

        # --- Encoder ---
        # Convert source indices to embeddings
        inputEmbedding_w = self.embeddingEncoder(source_input)

        # Linear projection for convolutional blocks
        inputEncoder = self.linearEncoderInput(inputEmbedding_w)

        # Pass through encoder convolutional blocks
        encoderOutput_z=self.encoderBlocks(inputEncoder)

        # Linear projection back to embedding space
        encoderOutput_z=self.linearEncoderOutput(encoderOutput_z)

        # Residual scaling for stabilization (Gehring 2017)
        scale = 1.0 / (2.0 * self.attentionLayer)
        encoderOutput_z = encoderOutput_z * scale + encoderOutput_z.detach() * (1 - scale)

        # Conditional input for decoder: combines encoder output with initial embeddings
        c_inputEncoder= (inputEmbedding_w+encoderOutput_z)*math.sqrt(0.5)

        # --- Decoder ---
        # Convert target indices to embeddings
        targetEmbedding_g = self.embeddingDecoder(target_input)

        # Linear projection of decoder embeddings
        inputDecoder = self.linearDecoderInput(targetEmbedding_g)

        # Pass through decoder blocks with attention
        decoder_attentionOutput = self.decoder_attentionBlocks(
            (inputDecoder, targetEmbedding_g, encoderOutput_z, c_inputEncoder)
        )

        # Final classification: project hidden states to logits over target vocabulary
        output=self.classification(decoder_attentionOutput)
        return output




