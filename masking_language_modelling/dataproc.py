
import torch
from torch.utils.data import Dataset
from collections import Counter
from itertools import chain
from typing import List, Optional

class MLMDataset(Dataset):
    def __init__(self, text_fpath: str, 
                 max_seq_len: int, 
                 mask_ratio:float = 0.15
                ):
        self.max_seq_len = max_seq_len
        self.mask_ratio = mask_ratio
        with open(text_fpath, 'r') as f:
            self.lines = f.readlines()
        
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]
    
import string
import re

class Tokenizer:
    punctuation = string.punctuation.replace('-', '')
    def __init__(self, 
                 max_vocab_size: int,
                 truncation: bool = True,
                 max_seq_len: Optional[int] = None,
                 padding: bool = True,
                ):
        self.pad_token = '<PAD>'
        self.mask_token = '<MASK>'
        self.cls_token = '<CLS>'
        self.sep_token = '<SEP>'
        self.special_tokens = [self.cls_token, self.sep_token, self.pad_token, self.mask_token]
        self.max_vocab_size = max_vocab_size
        self.truncation = truncation
        self.padding = padding
        
        if self.padding or self.truncation:
            assert not(max_seq_len is None)
            self.max_seq_len = max_seq_len
    
    def fit(self, dataset):
        most_common_words = Counter(chain.from_iterable(map(self._preproc, dataset)))\
                                                     .most_common(self.max_vocab_size-len(self.special_tokens))
        most_common_words = list(map(lambda x: x[0], most_common_words))
        self.vocab = dict(map(lambda x: (x[1], x[0]), enumerate(self.special_tokens + most_common_words)))
        return self
    
    def apply(self, text: str):
        pad_token_idx = self.vocab[self.pad_token]
        input_seq = self._preproc(text)[:self.max_seq_len]
        payload_tokens = list(map(lambda x: self.vocab.get(x, pad_token_idx), input_seq))
        padding_tokens = [pad_token_idx]*(self.max_seq_len-len(payload_tokens))
        return [self.vocab[self.cls_token]] + payload_tokens + padding_tokens
    
    def _preproc(self, text: str) -> List[str]:
        sub_pattern = f'[{re.escape(Tokenizer.punctuation)}]'
        return re.sub(sub_pattern, ' ', text.lower()).split()
    

def spawn_collate_fn(tokenizer, mask_ratio=0.15):
    cls_token_id = tokenizer.vocab[tokenizer.cls_token]
    sep_token_id = tokenizer.vocab[tokenizer.sep_token]
    mask_token_id = tokenizer.vocab[tokenizer.mask_token]
    
    def mask_objective(batch_token_ids, mask_ratio):
            masked_tokens = torch.rand(batch_token_ids.shape)<mask_ratio
            mask_arr = masked_tokens * (batch_token_ids != cls_token_id) * (batch_token_ids != sep_token_id)
            return mask_arr
        
    def custom_collate_fn(batch):
        input_ids = torch.Tensor(batch).long()
        mlm_mask = mask_objective(input_ids, mask_ratio)
        masked_input_ids = torch.where(mlm_mask, mask_token_id, input_ids).long()

        return {
            'input_tokens': input_ids,
            'masked_input_tokens': masked_input_ids,
#             'attention_mask': 1, # is this the same as mlm mask?
            'mlm_mask': mlm_mask,
        }
    return custom_collate_fn
