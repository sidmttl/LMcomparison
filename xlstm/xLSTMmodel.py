import wandb
import torch
import torch.nn as nn
from torch.nn import functional as F
from math import *
from xlstm.xLSTM import xLSTM as xlstm


# batch_size = 64 # how many independent sequences will we process in parallel?
# block_size = 256 # what is the maximum context length for predictions?
# max_iters = 5000
# eval_interval = 500
# learning_rate = 3e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 200
# n_embd = 384
# n_head = 6
# n_layer = 6
# dropout = 0.2

class XLSTMLanguageModel(nn.Module):

    def __init__(self, vocab_size, x, batch_size = 64, block_size = 256,
                max_iters = 5000, eval_interval = 500, learning_rate = 3e-4, 
                device = 'cuda' if torch.cuda.is_available() else 'cpu',
                eval_iters = 200, layers = ['s', 'm'], dropout = 0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = x.shape[2]
        self.block_size = x.shape[1]

        self.device = device
        self.layers = layers
        
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embd)
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embd)
        
        self.xlstm = xlstm(self.layers, x)
        
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.head = nn.Linear(self.n_embd, self.vocab_size)

        wandb.init(
            # set the wandb project where this run will be logged
            project="language-model",
            name="xlstm-based-transformers",
            # track hyperparameters and run metadata
            config={
                "batch_size": batch_size,
                "block_size": self.block_size,
                "learning_rate": learning_rate,
                "architecture": "xLSTM based transformers",
                "dataset": "tiny-shakespeare",
                "n_embd": self.n_embd,
                # "n_layer": self.n_layer,
                "dropout": dropout,
            }
        )
            
    def init_states(self, x):
        self.xlstm.init_states(x)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # T, C

        x = tok_emb + pos_emb # (B, T, C)

        x = self.xlstm(x)

        x = self.ln_f(x)

        logits = self.head(x)
        
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last self.block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx, idx_next
