import torch
from torch import nn
import torch.nn.functional as F

from Model_New.LayerModules_new import LinearTransformation_New


class Classification_New(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, target_vocab_size, p_dropout):
        super().__init__()
        self.linear = LinearTransformation_New(False, hidden_dim, embedding_dim, p_dropout)
        self.dropout = nn.Dropout(p_dropout)
        self.softmax_linear = LinearTransformation_New(True, embedding_dim, target_vocab_size, p_dropout)

    def forward(self, input):

        input = self.linear(input[0])
        input = self.dropout(input)

        logits = self.softmax_linear(input)

        sz = logits.size()
        prediction = F.softmax(logits.view(sz[0] * sz[1], sz[2]), dim=1)

        return prediction, logits
