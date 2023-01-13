
import torch

from multihead_attention import MultiHeadAttention

class BaseEncoderModel(torch.nn.Module):
    def __init__(self, n_heads, emb_size, vdim=None, kdim=None):
        super().__init__()
        self.layers = torch.nn.Sequential(
            MultiHeadAttention(n_heads=n_heads, emb_size=emb_size),
            MultiHeadAttention(n_heads=n_heads, emb_size=emb_size),
        )
    def forward(self, X: torch.Tensor):
        return self.layers(X)


class MLMHead(torch.nn.Module):
    def __init__(self, model, vocab_size, mask_ratio=0.15):
        super().__init__()
        self.model = model
        out_features = self.model.layers[-1].ffn.inplace.out_features
        self.mlm_layer = torch.nn.Linear(out_features, vocab_size, bias=False)
        self.mask_ratio = mask_ratio
        
    def forward(self, X): 
        state = self.model(X)
        logits = self.mlm_layer(state)
        seq_len = logits.shape[1]
        objective_mask = MLMHead.mask_objective(seq_len, self.mask_ratio)
        result = torch.softmax(logits, axis=-1)
        return objective_mask, result
        
    @staticmethod
    def mask_objective(seq_len, mask_ratio = 0.15):
        masked_tokens = torch.rand(seq_len)<mask_ratio
        return masked_tokens
        
        
