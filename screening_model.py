'''
RobotScreener
'''

import torch
from torch.utils.data import IterableDataset, DataLoader

import transformers
from transformers import AdamW
from transformers import RobertaForSequenceClassification, RobertaTokenizer, PretrainedConfig


device = torch.device("cpu") #torch.device('cuda')


def train(dataset: IterableDataset) -> bool:
    tokenizer = RobertaTokenizer.from_pretrained("allenai/biomed_roberta_base") 
    model     = RobertaForSequenceClassification.from_pretrained("allenai/biomed_roberta_base", 
                                                                 num_labels=n_labels).to(device=device) 
    
    
    optimizer = AdamW(model.parameters())


def train_and_save(titles, abstracts, labels, uuid):
    pass


class SRDataset():

    def __init__(self, titles, abstracts, labels): 
        super(SRDataset).__init__()
        self.titles = titles
        self.abstracts = abstracts
        self.labels = labels 
        self.N = len(self.titles)


    def __len__(self):
        return self.N

    def __getitem__(self, index):
        ti = self.titles[index]
        abstract = self.abstracts[index]
        y = self.labels[index]

        X = [ti + " [sep] " + abstract]
        return X, y



