import torch
from torch import nn



class InitialEmbedding_New(nn.Module):

    #TO-DO: Check Pre-Trained embeddings https://huggingface.co/docs/transformers/en/fast_tokenizers?tokenizer-classes=model-specific+tokenizer

    def __init__(self, vocab_size, fixedNumberOfInputElements, embedding_dim, p_dropout, hidden_dim, unknown_token):
        super().__init__()
        self.positional_Embedding = Embedding(vocab_size, fixedNumberOfInputElements, embedding_dim, unknown_token)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, input):
        input = self.positional_Embedding(input)
        input = self.dropout(input)
        return input


class Embedding(nn.Module):
    def  __init__(self, vocab_size, fixedNumberOfInputElements, embedding_dim, unknown_token):
        super(Embedding, self).__init__()
        self.embedding_vocabulary = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        torch.nn.init.normal_(self.embedding_vocabulary.weight, mean=0.0, std=0.1)

        self.embedding_positional = nn.Embedding(num_embeddings=fixedNumberOfInputElements, embedding_dim=embedding_dim)
        torch.nn.init.normal_(self.embedding_positional.weight, mean=0.0, std=0.1)
        self.unknown_token = unknown_token
        self.vocab_size = vocab_size

    def forward(self, input):
        batch_size, seq_len = input.size()

        # Create a mask for unknown tokens
        unknown_mask = (input >= self.vocab_size) | (input < 0)  # Identify invalid indices

        # Create a safe version of the input where invalid indices are replaced with unknown_token
        safe_input = input.clone()
        safe_input[unknown_mask] = self.unknown_token

        # Get word embeddings for all positions and all batches at once
        # Shape: (batch_size, seq_len, embedding_dim)
        word_embeddings = self.embedding_vocabulary(safe_input)

        # Generate positional embeddings for all positions at once
        positions = torch.arange(0, seq_len, device=input.device).unsqueeze(0).expand(batch_size, -1)
        positional_embeddings = self.embedding_positional(positions)

        # Add word and positional embeddings
        output_tensor = word_embeddings + positional_embeddings

        # Log if unknown tokens were used
        if unknown_mask.any():
            print(f"Used unknown token for {unknown_mask.sum().item()} positions")

        return output_tensor


