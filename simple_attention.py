
import torch
from torch import nn


class Attention(torch.nn.Module):
    def __init__(self, emb_size, kdim=None, vdim=None):
        """
        emb_size - size of input embedding vector
        by default w_size(aka k_dim, v_dim equal to emb_size) in https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention
        
        """
        super().__init__()
        self.emb_size  = emb_size
        
        # Inner dimensions of Q and K matrices are equal, dimension of V might be different. 
        # ...but by deafault all of them are equal to emb_size
        self.kdim  = kdim or emb_size
        self.vdim  = vdim or emb_size
#         self.W_Q = nn.Parameter(torch.Tensor(self.emb_size, self.kdim))
#         self.W_K = nn.Parameter(torch.Tensor(self.emb_size, self.kdim))
#         self.W_V = nn.Parameter(torch.Tensor(self.emb_size, self.vdim))
        self.W_Q = nn.Linear(self.emb_size, self.kdim)
        self.W_K = nn.Linear(self.emb_size, self.kdim)
        self.W_V = nn.Linear(self.emb_size, self.vdim)
        
    def forward(self, X: torch.Tensor):
        """
        The first step is to calculate the Query, Key, and Value matrices. 
        We do that by packing our embeddings into a matrix X, and multiplying it by the weight 
        matrices we’ve trained (WQ, WK, WV).
        
        - Every row in the X matrix corresponds to a word in the input sentence. 
        
        """
#         Q = torch.matmul(X, self.W_Q)
#         K = torch.matmul(X, self.W_K)
#         V = torch.matmul(X, self.W_V)
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)
        numerator = torch.matmul(Q, torch.transpose(K, 1, 2))
        denominator = (self.kdim)**0.5
        
        fraction = torch.softmax(numerator/denominator, axis=-1)
        self_attention = torch.matmul(fraction, V)
        return self_attention
