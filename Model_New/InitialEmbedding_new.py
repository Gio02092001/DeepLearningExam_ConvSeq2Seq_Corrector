import torch
from torch import nn

class InitialEmbedding_New(nn.Module):
    """
       Wrapper around the custom Embedding layer.
       Combines token embeddings with positional embeddings, then applies dropout.
    """

    def __init__(self, vocab_size, fixedNumberOfInputElements, embedding_dim, p_dropout, hidden_dim, unknown_token):
        super().__init__()
        # Custom embedding layer (word + positional embeddings)
        self.positional_Embedding = Embedding(vocab_size, fixedNumberOfInputElements, embedding_dim, unknown_token)

        # Dropout for regularization
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, input):
        """
            Args:
                input: Tensor of shape [batch_size, seq_len] with token indices

            Returns:
                input: Tensor of shape [batch_size, seq_len, embedding_dim]
        """
        input = self.positional_Embedding(input) # word + positional embeddings
        input = self.dropout(input) # apply dropout
        return input


class Embedding(nn.Module):
    """
        Custom embedding layer that combines:
          - Word embeddings (lookup by token index)
          - Positional embeddings (lookup by sequence position)
    """

    def  __init__(self, vocab_size, fixedNumberOfInputElements, embedding_dim, unknown_token):
        super(Embedding, self).__init__()

        # Embedding for vocabulary tokens
        self.embedding_vocabulary = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        torch.nn.init.normal_(self.embedding_vocabulary.weight, mean=0.0, std=0.1)

        # Embedding for positional indices
        self.embedding_positional = nn.Embedding(num_embeddings=fixedNumberOfInputElements, embedding_dim=embedding_dim)
        torch.nn.init.normal_(self.embedding_positional.weight, mean=0.0, std=0.1)
        self.unknown_token = unknown_token
        self.vocab_size = vocab_size

    def forward(self, input):
        """
            Args:
                input: Tensor of shape [batch_size, seq_len] containing token indices

            Returns:
                output_tensor: Tensor of shape [batch_size, seq_len, embedding_dim]
                               word embeddings + positional embeddings
        """
        batch_size, seq_len = input.size()

        # Lookup word embeddings
        word_embeddings = self.embedding_vocabulary(input)

        # Lookup positional embeddings for positions [0...seq_len-1]
        positions = torch.arange(0, seq_len, device=input.device).unsqueeze(0).expand(batch_size, -1)
        positional_embeddings = self.embedding_positional(positions)

        # Add word and positional embeddings
        output_tensor = word_embeddings + positional_embeddings

        return output_tensor