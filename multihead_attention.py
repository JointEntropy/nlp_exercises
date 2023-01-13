
from simple_attention import Attention
import torch

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_heads, emb_size, vdim=None, kdim=None):
        super().__init__()
        self.n_heads = n_heads
        self.emb_size  = emb_size
        self.kdim  = kdim or emb_size
        self.vdim  = vdim or emb_size
        assert (self.vdim % self.n_heads)==0, 'emb_size must be divisible by n_heads'
#         self.W_0 = nn.Parameter(torch.Tensor(self.emb_size, self.vdim))
        self.W_0 = torch.nn.Linear(self.vdim, self.vdim, bias=False)
        self.heads = [Attention(self.emb_size, vdim=self.vdim//self.n_heads, 
                                kdim=self.kdim) for i in range(self.n_heads)]
        self.ffn = torch.nn.ReLU(torch.nn.Linear(self.vdim, self.vdim))
        
    def forward(self, X: torch.Tensor):
        attentions = []
        # TODO this can and should be done in parallel
        for head in self.heads:
            z_i = head(X)
            attentions.append(z_i)
        Z = torch.concatenate(attentions, axis=-1)
        multihead_attention = self.W_0(Z)
        output = self.ffn(multihead_attention)
        return output
