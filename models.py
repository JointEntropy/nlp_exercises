# coding: utf-8
from datasets import load_dataset
tmp_dataset = load_dataset("rotten_tomatoes")
# with open('rt.txt', 'w') as f:
#     for x in tmp_dataset['train']:
#         if len(x['text'].split())<10:
#             continue
#         f.write(x['text']+'\n')
from datasets import load_dataset
tmp_dataset = load_dataset("rotten_tomatoes")
# with open('rt.txt', 'w') as f:
#     for x in tmp_dataset['train']:
#         if len(x['text'].split())<10:
#             continue
#         f.write(x['text']+'\n')
tmp_dataset['train']
tmp_dataset['train'][0]
# with open('rt.txt', 'w') as f:
#     for x in tmp_dataset['train']:
#         if len(x['text'].split())<10:
#             continue
#         f.write(x['text']+'\n')
raw_df = pd.DataFrame(tmp_dataset['train'])
raw_df
import pandas as pd
# with open('rt.txt', 'w') as f:
#     for x in tmp_dataset['train']:
#         if len(x['text'].split())<10:
#             continue
#         f.write(x['text']+'\n')
raw_df = pd.DataFrame(tmp_dataset['train'])
raw_df
tmp_dataset
tmp_dataset['validation']
