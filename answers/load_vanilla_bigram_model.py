import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C) -> (4,8,65)
        B, T, C = logits.shape
        logits = logits.view(-1, C) # (B,T,C) -> (B*T,C)
        targets = targets.view(B*T) # (B,T) -> (B*T)
        loss = F.cross_entropy(logits, targets)
        return logits

m = BigramLanguageModel()
logits, loss = m(xb, yb)
print(logits.shape)
