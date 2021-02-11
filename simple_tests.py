import numpy as np 

import pandas as pd 

import screening_model
from screening_model import SRDataset

import torch
from torch.utils.data import IterableDataset, DataLoader

def train_from_csv(review_path="example-datasets/labels_73.csv") -> None:
    df = pd.read_csv(review_path)
    df = df.replace(np.nan, "") 

    # massage the labels to be binary only (0/1); first replace
    # extant `0' values with `1' since these are maybes, then map
    # `-1's to `0'. 
    labels = df["level1_labels"].values  
    labels = np.where(labels == 0, 1, labels)
    labels = np.where(labels == -1, 0, labels)

    dataset = SRDataset(df["title"].values[:100], 
                        df["abstract"].values[:100], 
                        labels[:100])
    
    success = screening_model.train_and_save(dataset, "1337") 
    print(success)
    #dl = DataLoader(dataset, batch_size=16)
    #m = screening_model.train(dl)


train_from_csv()