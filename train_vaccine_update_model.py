import pandas as pd 
import numpy as np 

import screening_model
from screening_model import SRDataset

import torch
#from torch.utils.data import IterableDataset, DataLoader

def train(csv_path:str ="example-datasets/covid-vaccines.csv", val_split:float = 0.2):
    df = pd.read_csv(csv_path)

    ''' pull out/collapse the labels '''
    labelers = ['KDanko', 'brown_2021', 'mbhuma', 'gadam']
    df_labels = df[labelers].dropna().replace("o", 0).astype(int)
    # collapse via 'max' over labelers
    labels = df_labels.max(axis=1).values 

    n_dropped = df.shape[0] - df_labels.shape[0]

    ''' titles/abstracts ''' 
    titles    = df['title'].values[:-n_dropped] 
    abstracts = df['abstract'].replace(np.nan, "").values[:-n_dropped]
    
    N = len(titles)
    val_size = int(val_split * N)

    # randomly split into trian/test
    train_indices = np.random.choice(N, size=N - val_size, replace=False)
    train_mask = np.zeros(N, dtype=bool)
    train_mask[train_indices] = True 
    train_titles    = titles[train_mask]
    train_abstracts = abstracts[train_mask]
    train_labels    = labels[train_mask]

    val_titles      = titles[~train_mask]
    val_abstracts   = abstracts[~train_mask]
    val_labels      = labels[~train_mask]

    vax_dataset_tr  = SRDataset(train_titles, train_abstracts, train_labels) 
    vax_dataset_val = SRDataset(val_titles, val_abstracts, val_labels)

    screening_model.train_and_save(vax_dataset_tr, 1, val_dataset=vax_dataset_val)

train()
