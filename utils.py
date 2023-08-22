import faiss
import logging
import random
import numpy as np
from typing import Tuple
import torch
from torch.utils.data import Dataset, Sampler
from pytorch_metric_learning import losses
from torch import nn
import torch.nn.functional as F
from collections import defaultdict
import math
import visualizations


# Compute R@1, R@5, R@10, R@20
RECALL_VALUES = [1, 5, 10, 20]

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class CosPlace(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gem = GeM()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = self.gem(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)

class MixVPR(nn.Module):
    def __init__(self, in_channels=512, in_h=7, in_w=7, out_channels=512, mix_depth=4, mlp_ratio=1, out_rows=4):
        super().__init__()
        self.in_h = in_h 
        self.in_w = in_w
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_rows = out_rows
        self.mix_depth = mix_depth
        self.mlp_ratio = mlp_ratio
        hw = in_h * in_w
        self.mix = nn.Sequential(*[
            FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio)
            for _ in range(self.mix_depth)
        ])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x):
        x = x.flatten(2) #collasso ultima dimensione -> riga*colonna
        x = self.mix(x)
        x = x.permute(0, 2, 1) #inverto dimensioni
        x = self.channel_proj(x) #feature dim reduction col
        x = x.permute(0, 2, 1) #inverto dimensioni
        x = self.row_proj(x) #feature dim reduction col row
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        return x

class ProxyHead(nn.Module):
    def __init__(self, in_dim, out_dim = 512):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.fc(x)
        x = F.normalize(x, p=2)
        return x

class Proxy:
    def __init__(self, tensor = None, n = 1, dim = 512):
        if tensor is None:
            self.__arch = torch.zeros(dim)
        else:
            self.__arch = tensor
        self.__n = n

    def get_avg(self):
        return self.__arch / self.__n

    def __add__(self, other):
        return ProxyBank.Proxy(tensor=self.__arch + other.__arch, n=self.__n + other.__n)

class ProxyBank:
    def __init__(self, proxy_dim = 512):
        self.__bank = {}
        self.proxy_dim = proxy_dim
        self.__base_index = faiss.IndexFlatL2(self.proxy_dim)
        self.index = faiss.IndexIDMap(self.__base_index)

    def update_bank(self, proxies, labels):
        #riempo la banca
        for d, l in zip(proxies, labels):
            # Create or Update the bank
            if l.item not in self.__bank:
                self.__bank[l.item()] = Proxy(d)
            else:
                self.__bank[l.item()] = self.__bank[l.item()] + Proxy(d)

    def update_index(self):
        self.index.reset()
        for label, proxy in self.__bank.items():
            self.index.add_with_ids(proxy.get_avg().reshape(1,-1).detach().cpu() , label)
    
    # Empty all the dictionaries and indeces created so far
    def reset(self):
         del self.__bank
         del self.index
         self.__bank = {}
         self.__base_index = faiss.IndexFlatL2( self.proxy_dim )
         self.index = faiss.IndexIDMap( self.__base_index )
    
    def batch_sampling(self , batch_dim):
        batches = []
        while len(self.__bank) >= batch_dim:
            # Extract a label
            rand_index = random.randint(0 , len(self.__bank) - 1)
            rand_bank = list(self.__bank.items())[rand_index]
            # From ProxyAccumulator to the Proxy, using get_avg()
            start_proxy = rand_bank[1].get_avg()
            # Compute the kNN with faiss_index
            _, batch = self.index.search(start_proxy.reshape(1,-1).detach().cpu(), batch_dim)
            # From [[]] to []: only one row needed
            batch = batch[0]
            # adding it to the list of batches used in the sampler
            batches.append(batch)
            for label in batch:
                del self.__bank[label]
            self.index.remove_ids(batch)
        self.reset()
        return batches 
       
class ProxyBankBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, bank):
        self.is_first_epoch = True
        self.dataset = dataset
        self.batch_size = batch_size
        self.iterable_size = len(dataset) // batch_size
        self.bank = bank
        self.batch_iterable = []
        self.counter = 0
    def __iter__(self): 
        if self.is_first_epoch and self.counter % 2 == 0:
            self.is_first_epoch = False
            random_indeces_perm = torch.randperm(len(self.dataset))
            indeces =  torch.split(random_indeces_perm , self.batch_size)
            self.batch_iterable = iter(indeces)
        elif self.counter % 2 == 0:
            indeces = self.bank.batch_sampling(self.batch_size)
            self.batch_iterable = iter(indeces)
        self.counter += 1
        return  self.batch_iterable
    
    def __len__(self):
        return self.iterable_size

def compute_recalls(eval_ds: Dataset, queries_descriptors : np.ndarray, database_descriptors : np.ndarray,
                    output_folder : str = None, num_preds_to_save : int = 0,
                    save_only_wrong_preds : bool = True) -> Tuple[np.ndarray, str]:
    """Compute the recalls given the queries and database descriptors. The dataset is needed to know the ground truth
    positives for each query."""

    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(queries_descriptors.shape[1])
    faiss_index.add(database_descriptors)
    del database_descriptors

    logging.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_descriptors, max(RECALL_VALUES))

    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    recalls = np.zeros(len(RECALL_VALUES))
    for query_index, preds in enumerate(predictions):
        for i, n in enumerate(RECALL_VALUES):
            if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by queries_num and multiply by 100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls)])

    # Save visualizations of predictions
    if num_preds_to_save != 0:
        # For each query save num_preds_to_save predictions
        visualizations.save_preds(predictions[:, :num_preds_to_save], eval_ds, output_folder, save_only_wrong_preds)
    
    return recalls, recalls_str
