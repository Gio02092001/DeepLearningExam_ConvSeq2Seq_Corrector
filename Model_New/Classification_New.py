from torch import nn
import torch.nn.functional as F
from Model_New.LayerModules_new import LinearTransformation_New

class Classification_New(nn.Module):
    """
        Classification module for decoder outputs.

        Steps:
        1. Project hidden states into embedding space.
        2. Apply dropout for regularization.
        3. Map embeddings to target vocabulary logits.
        4. Apply softmax to obtain prediction probabilities.
    """

    def __init__(self, hidden_dim, embedding_dim, target_vocab_size, p_dropout):
        """
            Initialize classification module.

            Args:
                hidden_dim (int): Dimensionality of hidden states.
                embedding_dim (int): Dimensionality of intermediate embedding.
                target_vocab_size (int): Number of classes (vocabulary size).
                dropout_prob (float): Dropout probability.
        """
        super().__init__()
        # Project hidden states into embedding space
        self.linear = LinearTransformation_New(False, hidden_dim, embedding_dim, p_dropout)

        # Dropout layer
        self.dropout = nn.Dropout(p_dropout)

        # Map embeddings to target vocabulary logits
        self.softmax_linear = LinearTransformation_New(True, embedding_dim, target_vocab_size, p_dropout)

    def forward(self, input):
        """
            Forward pass for classification.

            Args:
                input (tuple): Tuple containing decoder outputs. We use the first element [batch, seq_len, hidden_dim].

            Returns:
                prediction (Tensor): Softmax probabilities over target classes [batch*seq_len, target_vocab_size].
                logits (Tensor): Raw logits before softmax [batch, seq_len, target_vocab_size].
        """
        # Project decoder outputs to embedding space
        input = self.linear(input[0])

        # Apply dropout
        input = self.dropout(input)

        # Map embeddings to target vocabulary logits
        logits = self.softmax_linear(input)

        # Flatten batch and sequence dimensions for softmax
        sz = logits.size()
        prediction = F.softmax(logits.view(sz[0] * sz[1], sz[2]), dim=1)

        return prediction, logits