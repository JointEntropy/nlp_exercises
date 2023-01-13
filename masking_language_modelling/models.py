
import torch

from multihead_attention import MultiHeadAttention

class BaseEncoderModel(torch.nn.Module):
    def __init__(self, vocab_size, n_heads, emb_size, vdim=None, kdim=None, padding_idx=None):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx),
            MultiHeadAttention(n_heads=n_heads, emb_size=emb_size),
            MultiHeadAttention(n_heads=n_heads, emb_size=emb_size),
        )
    def forward(self, X: torch.Tensor):
        return self.layers(X)


class MLMHead(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        out_features = self.model.layers[-1].ffn.inplace.out_features
        vocab_size = self.model.layers[0].num_embeddings
        self.mlm_layer = torch.nn.Linear(out_features, vocab_size, bias=False)
        
    def forward(self, X: torch.Tensor): #, masked_tokens: torch.Tensor): 
        state = self.model(X)
        logits = self.mlm_layer(state)
        # you can't truncate only masked tokens here because the shape of the bantch will be broken.
        #         masked_tokens_logits = logits[:, masked_tokens, :]
#         result = torch.softmax(logits, axis=-1) # will add this in loss
        return logits
