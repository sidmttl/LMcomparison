import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum
import math

class ResidualBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):

        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.mixer = MambaBlock(d_model, d_state, d_conv, expand)
        self.norm = RMSNorm(d_model)
        

    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
            
            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
            
        """
        output = self.mixer(self.norm(x)) + x

        return output

class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state = 16, d_conv=4, expand = 2, dt_rank="auto", bias = False, conv_bias = True):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1].
        d_model = embed_dim

        """
        super().__init__()
        self.d_model = d_model
        self.dt_rank = math.ceil(self.d_model / 16)
        self.d_inner = int(expand * self.d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        

    def forward(self, x):
       
        (b, l, d) = x.shape
        
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)

        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        output = self.out_proj(y)

        return output

    
    def ssm(self, x):
       
        (d_in, n) = self.A_log.shape
        
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)
        
        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        
        y = self.selective_scan(x, delta, A, B, C, D) 
        
        return y

    
    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1) 
        
        y = y + u * D
    
        return y


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
        