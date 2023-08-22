import faiss
import logging
import numpy as np
from typing import Tuple
import torch
from torch.utils.data import Dataset, Sampler
from pytorch_metric_learning import losses
from torch import nn
import torch.nn.functional as F
from collections import defaultdict
import math

from torchmetrics import RetrievalRecall

import visualizations


# Compute R@1, R@5, R@10, R@20
RECALL_VALUES = [1, 5, 10, 20]

class ContrastiveTransformation:
    def __init__(self, customized_transforms, n_views=2):
        self.customized_transforms = customized_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.customized_transforms(x) for i in range(self.n_views)]
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
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)


class MixVPR(nn.Module):
    def __init__(self,
                 in_channels=3,
                 in_h=224,
                 in_w=224,
                 out_channels=3,
                 mix_depth=1,
                 mlp_ratio=1,
                 out_rows=4,
                 ) -> None:
        super().__init__()

        self.in_h = in_h # height of input feature maps
        self.in_w = in_w # width of input feature maps
        self.in_channels = in_channels # depth of input feature maps
        
        self.out_channels = out_channels # depth wise projection dimension
        self.out_rows = out_rows # row wise projection dimesion

        self.mix_depth = mix_depth # L the number of stacked FeatureMixers
        self.mlp_ratio = mlp_ratio # ratio of the mid projection layer in the mixer block

        hw = in_h*in_w
        self.mix = nn.Sequential(*[
            FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio)
            for _ in range(self.mix_depth)
        ])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x):
        x = x.flatten(2)
        x = self.mix(x)
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        return x
class ProxyHead(nn.Module):
    def __init__(self, in_dim, out_dim = 128):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.loss_fn = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=0.001)
    
    def forward(self, x):
        x = self.fc(x)
        x = F.normalize(x, p=2)
        return x
    
    def fit(self, descriptors, labels):
        compressed_descriptors = self(descriptors)
        loss = self.loss_fn(compressed_descriptors, labels)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

class ProxyBank:
    """This class stores the places' proxies together with their identifier
       and performs exhaustive search on the index to retrieve the mini-batch sampling pool."""

    def __init__(self, k = 10, bank = None):
        if bank is None:
            bank = defaultdict(ProxyBank.Proxy)
        self.__bank = bank
        self.k = k
        self.dim = 128
        self.n_samples = self.dim
        self.index = faiss.index_factory(self.dim, 'IDMap,Flat')

    def update_bank(self, proxies, labels):
        #riempo la banca
        for d, l in zip(proxies, labels):
            self.__bank[l.item()] = self.__bank[l.item()] + ProxyBank.Proxy(d)
        #lista numpy [0-128 (dim)]
        idx = np.arange(self.n_samples)
        #da [[], []] a numero, che corrisponde alla media di quei proxy. Guarda metodo get_avg nella classe proxy  
        proxies_by_idx = torch.stack([self.__bank[i].get_avg() for i in idx]) 
        #aggiungo ad index descriptor e suo id
        self.index.add_with_ids(proxies_by_idx, idx) 

    def build_index(self):
        bank_values = self.__bank.values()
        bank_values = list(map(lambda bank_value: bank_value.get_avg(), bank_values))
        bank_values = np.vstack(bank_values)

        labels = np.array(list(self.__bank.keys()))

        s = {}
        for e in range(self.n_samples):
            s[e] = 1

        ids = []
        #prelevo un id a caso
        list_idx = np.arange(self.n_samples)
        np.random.shuffle(list_idx)
        #se id i non è ancora stato slezionato
        for i in list_idx:
            if s[i] == 1: 
                # ritorna distanza, [[]] contente id (not placeID, ma quelli dell'index) => quindi prendo id[0] 
                # per eliminare [] più esterna, tanto dim = 1
                _, id = self.index.search(bank_values[i:i + 1], self.k)
                id = id[0]
                #aggiungo a ids gli ultimi id estratti []
                ids.extend(id)
                self.index.remove_ids(id)
                for e in id:
                    #id già selezionato
                    s[e] = 0

        ids = ids[:bank_values.shape[0]]
        return labels[ids]

    class Proxy:
        def __init__(self, tensor = None, n = 1, dim = 128):
            if tensor is None:
                self.__arch = torch.zeros(dim)
            else:
                self.__arch = tensor
            self.__n = n

        def get_avg(self):
            return self.__arch / self.__n

        def __add__(self, other):
            return ProxyBank.Proxy(tensor=self.__arch + other.__arch, n=self.__n + other.__n)
class ProxySampler(Sampler):
    def __init__(self, indexes_list, batch_size = 10):
        self.batch_size = batch_size
        self.batches = [indexes_list[batch_size * i: batch_size * (i + 1)] for i in
                        range(math.ceil(len(indexes_list) / batch_size))]

    def __iter__(self):
        for batch in self.batches:
            print("batch:", batch)
            yield batch

    def __len__(self):
        return len(self.batches)


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
    
    #print(f"Predictions: {predictions[:2]} shape: {predictions.shape}")
    #print(f"Positives_per Query: {positives_per_query[:2]} shape: {positives_per_query.shape}")
    #print(f"Evaluation ds query num: {eval_ds.queries_num}")
    
    recalls = np.zeros(len(RECALL_VALUES))
    for query_index, preds in enumerate(predictions):
        for i, n in enumerate(RECALL_VALUES):
            if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                #print("\nBINGO\n")
                recalls[i:] += 1
                break
    # Divide by queries_num and multiply by 100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    #print(f"\nEval_querin num {eval_ds.queries_num}\n")
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls)])

   # print(recalls_str)
    # Save visualizations of predictions
    if num_preds_to_save != 0:
        # For each query save num_preds_to_save predictions
        visualizations.save_preds(predictions[:, :num_preds_to_save], eval_ds, output_folder, save_only_wrong_preds)
    
    return recalls, recalls_str
