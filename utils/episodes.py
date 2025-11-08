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
    families = {}
    for path in glob.glob(os.path.join(encoded_dir, "*.pt")):
        pack = torch.load(path)

        #Handle two possible save formats:
        # - dict: {"X": Tensor[N,L]...}
        # X -> data sensor, N -> number of sequences, L -> sequence length (columns)
        # - list: [Tensor[L], Tensor[L], ...]
        if isinstance(pack, dict): #saviing under new format: with the main tensor under key "X"
            X = pack.get("X", None)
        else:
            #if its a legacy list of 1D tensors
            if len(pack) == 0:
                continue
            X = torch.stack(pack)
        
        #Basic validity checks
        if X is None or not isinstance(X, torch.Tensor):
            continue
        if X.ndim != 2 or X.dtype != torch.long: #ensures your encoder can embed these (embedding requires long IDs)
            continue
        if X.shape[0] == 0:
            continue
            
        fam = os.path.splitext(os.path.basename(path))[0] #strip folder and .pt to get the family name
        families[fam] = {"X":X.long()} #ensure dtype is long (embedding expects longs)
    return families
#Sample output:
#families = {
#    "kinase": {"X": Tensor[N_seq, L]},
#    "transferase": {"X": Tensor[N_seq, L]},...}

#Episode sampler - every time you call sample_episode():
# - randomly chooses N families (classes)
# - within each family, randomly picks K support and Q query sequences
# - returns 4 tensors: support_x, support_y, query_x, query_y

class EpisodeSampler:
    '''
    Samples N-way K-shot episodes:
    - support_x: (N*K, L), support_y: (N*K, ) labels 0..N-1
    - query_x: (N*Q, L), query_y: (N*Q, ) labels 0..N-1 

    # (N*K, L) means its a 2D tensor
    # (N*K, ) means its a 1D tensor -> comma just emphasizes this is a flat vector
    # query_y and support_y are labels for each support and query sequence
    '''
    def __init__(self, families: Dict[str, Dict[str, torch.Tensor]], 
                 N: int = 5, K: int = 5, Q: int = 10, device: str = "cpu"):
        self.N = N
        self.K = K
        self.Q = Q
        self.device = device

        #keep only families that have at least K+Q sequences
        self.names = []
        for name, pack in families.items(): #pack is the dictionary holding your tensor
            if pack['X'].shape[0] >= (K+Q): #if a family has at least K+Q episodes, then it qualifies to use in an episode
                self.names.append(name)
        if len(self.names) < N: #check to validate if we have enough names for the episode
            raise ValueError( #returns error message if not
                f"Need at least {N} families with >= {K+Q} sequences each; found {len(self.names)}."
            )
        #Store only valid ones to avoid extra dict lookups
        self.families = {}
        for name in self.names:
            self.families[name] = families[name]
    
    def sample_episode(self):
        '''
        Randomly pick N families, then sample K support and Q query from each.
        Returns:
            support_x: Tensor (N*K, L) -> 2D tensor
            support_y: Tensor (N*K,)  values in [0..N-1] -> 1D tensor
            query_x:   Tensor (N*Q, L) -> 2D tensor
            query_y:   Tensor (N*Q,) -> 1D tensor
        '''
        #Randomly choose N distinct families for this episode
        chosen = random.sample(self.names, self.N)

        support_x, support_y = [], []
        query_x, query_y = [], []

        for class_label, fam in enumerate(chosen):
            #shape [N_seq, L] (N_seq = number of sequences in this family; L = fixed length like 400).
            X = self.families[fam]["X"]

            # random permutation of indices; take first K+Q
            # creates a random permutation of all indices from 0 to N_seq - 1.
            perm = torch.randperm(X.shape[0])[: self.K + self.Q]
            #Split into K support and Q query
            s_idx = perm[: self.K] # keeps only the first (K + Q) indices,because we only need that many sequences for this episode (K support + Q query)
            #now split them into two groups: support, and query set    
            q_idx = perm[self.K : self.K + self.Q] #take the first K indices out of the shuffled lisrt to be used as support samples

            support_x.append(X[s_idx])
            query_x.append(X[q_idx])
            # makes a list of K identical labels for that class and .extend adds all of those labels tp support_y
            support_y.extend([class_label] * self.K)
            # Makes a list of Q copies of the current class label and adds them to query y
            query_y.extend([class_label] * self.Q)

            # Concatenate lists into big tensors and move to target device (cpu/mps )
            support_x = torch.cat(support_x, dim = 0).to(self.device) #(N*K, L)
            query_x = torch.cat(query_x, dim=0).to(self.device) #(N*Q, L)
            support_y = torch.tensor(support_y, dtype=torch.long, device=self.device)
            query_y = torch.tensor(query_y, dtype=torch.long, device=self.device)

            '''
                support_x = [Tensor_class0, Tensor_class1, ...]  # each of shape (K, L)
                query_x   = [Tensor_class0, Tensor_class1, ...]  # each of shape (Q, L)
                support_y = [0, 0, 0, 1, 1, 1, 2, ...]           # list of labels
                query_y   = [0, 0, 1, 1, 2, 2, ...]
            '''
            return support_x, support_y, query_x, query_y

    


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

