import wandb
import torch
import torch.nn as nn
from torch.nn import functional as F
from .mamba_block import ResidualBlock

class MambaLanguageModel(nn.Module):
    def __init__(self, vocab_size, batch_size = 64, block_size = 128, max_iters = 5000, eval_interval = 500, learning_rate = 3e-4, n_embd = 384, n_head = 6,
                n_layer = 6, dropout = 0.2, device = 'cpu'):
        super().__init__()
        self.batch_size = batch_size
        self.block_size = block_size
        self.learning_rate = learning_rate
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, self.n_embd)
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embd)
        self.blocks = nn.ModuleList([ResidualBlock(self.n_embd, d_state=16, d_conv=4, expand=2) for _ in range(self.n_layer)])
        self.lm_head = nn.Linear(self.n_embd, vocab_size)
        wandb.init(
        # set the wandb project where this run will be logged
        project = "language-model",
        name = "mamba-based-transformers",
        # track hyperparameters and run metadata
        config = {
            "batch_size": self.batch_size,
            "block_size": self.block_size,
            "learning_rate": self.learning_rate,
            "architecture": "Mamba based transformer LanguageModel",
            "dataset": "tiny-shakespeare",
            "n_embd": self.n_embd,
            "n_head": self.n_head,
            "n_layer": self.n_layer,
            "dropout": self.dropout,
            }
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx