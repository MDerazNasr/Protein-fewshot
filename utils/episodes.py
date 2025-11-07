#building episode sampler
#loaded_encoded_families - reads your data/encoded/*.pt files into memory
#each .pt is one protein family with a tensor of sequences (shape[N_seq, L], dtype long, where 0 is PAD)

import os, glob, random, torch
from typing import Dict

def loaded_encoded_families(encoded_dir: str) -> Dict[str, Dict[str, torch.Tensor]]:
    '''
    Load all family tensors from data/encoded/*.pt
    Returns: {family_name: {"X": Tensor[N, L]}}
    Skips empty/ malformed files safety
    '''
    # families = {}
    # for path in glob.glob(os.path.join(encoded_dir, "*.pt")):


#Episode sampler - every time you call sample_episode():
# - randomly chooses N families (classes)
# - within each family, randomly picks K support and Q query sequences
# - returns 4 tensors: support_x, support_y, query_x, query_y

'''
we create tiny tasks (episodes) on the fly; 
pick N classes,
take K labeled examples per class (suport)
Q unlabelled per class (query)
Train the model to classify queries by comparing to support prototypes

Why episodes?
Few-shot meta-learning doesn’t learn one big classifier. 
It learns how to quickly build a classifier from a handful of examples (K-shot) and generalize to new classes. 
So we train it on thousands of small supervised problems (episodes).
'''

