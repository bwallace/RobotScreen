'''
RobotScreener

Simple module for consuming labeled abstract screening data, training
a model on the basis of this, and dumping it to disk.
'''
import os 

from typing import Type, Tuple

import numpy as np 

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader, WeightedRandomSampler

import transformers
from transformers import AdamW
from transformers import RobertaForSequenceClassification, RobertaTokenizer, PretrainedConfig

###
# Globals; need to be edited for server.
import config
device = torch.device(config.device_str)
WEIGHTS_PATH = config.weights_path_str
###

def train(dl: DataLoader, epochs: int = 1) -> Tuple[Type[torch.nn.Module], Type[transformers.PreTrainedTokenizer]]:

    ''' Model and optimizer ''' 
    tokenizer = RobertaTokenizer.from_pretrained("allenai/biomed_roberta_base") 
    model     = RobertaForSequenceClassification.from_pretrained("allenai/biomed_roberta_base", 
                                                                 num_labels=2).to(device=device) 
    #import pdb; pdb.set_trace()
    for param in list(model.parameters())[-1:]:
        param.requires_grad = False
    
    optimizer = AdamW(model.parameters())

    best_val = np.inf
    for epoch in range(epochs):
        print(f"on epoch {epoch}.")
        model.train()
        running_losses = []
        
        for batch_num, (X, y) in enumerate(dl):

            optimizer.zero_grad()

            batch_X_tensor = tokenizer.batch_encode_plus(X, max_length=512, 
                                                        add_special_tokens=True, 
                                                        pad_to_max_length=True)
            batch_y_tensor = torch.tensor(y)
            model_outputs = model(torch.tensor(batch_X_tensor['input_ids']).to(device=device), 
                              attention_mask=torch.tensor(batch_X_tensor['attention_mask']).to(device=device), 
                              labels=batch_y_tensor.to(device=device))
          
            model_outputs['loss'].backward()
            running_losses.append(model_outputs['loss'].detach().float())
            if batch_num % 10 == 0:
                avg_loss = sum(running_losses[-10:])/len(running_losses[-10:])
                print(f"avg loss for last 10 batches: {avg_loss}")
            optimizer.step()

    return model, tokenizer
    
def eval_model(val_data: DataLoader, model: Type[torch.nn.Module], tokenizer: Type[transformers.PreTrainedTokenizer]):
    preds, labels = [], []
    with torch.no_grad():
        model.eval()
        for (X, y) in val_data:
            
            batch_X_tensor = tokenizer.batch_encode_plus(X, max_length=512, 
                                                        add_special_tokens=True, 
                                                        pad_to_max_length=True)
            model_outputs = model(torch.tensor(batch_X_tensor['input_ids']).to(device=device), 
                              attention_mask=torch.tensor(batch_X_tensor['attention_mask']).to(device=device))
            import pdb; pdb.set_trace()
            preds.extend(model_outputs['logits'].cpu().numpy().tolist())
            #preds.extend(model_outputs)
            labels.extend(y)

    import pdb; pdb.set_trace()
    return (preds, labels)


def get_weighted_sampler(dataset: Dataset) -> WeightedRandomSampler:
    # total number of positive instances
    n = dataset.labels.shape[0]
    n_pos = dataset.labels[dataset.labels>0].shape[0]
    n_neg = n - n_pos

    # split half the mass over the pos examples
    pos_weight = 0.5 / n_pos 
    neg_weight = 0.5 / n_neg 

    sample_weights = neg_weight * torch.ones(n, dtype=torch.float)
    pos_indices = np.argwhere(dataset.labels).squeeze()
    sample_weights[pos_indices] = pos_weight

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=n,
        replacement=True)

    return sampler 

def train_and_save(sr_dataset: Dataset, uuid: str, batch_size: int = 8, 
                    epochs: int = 1, val_dataset: Dataset = None) -> bool:
    '''
    Trains a classification model on the given review dataset and dumps
    to disk. If a val_dataset is provided, performance is evaluated on 
    this each epoch, and the best model is saved.
    '''

    # this is a sampler that assigns larger sampling weights to (rare) positive
    # examples for batch construction, to account for data imbalance.
    weighted_sampler = get_weighted_sampler(sr_dataset)
    
    dl = DataLoader(sr_dataset, batch_size=batch_size, sampler=weighted_sampler)
    model, tokenizer = train(dl, epochs=epochs)

    if val_dataset is not None: 
        val_dl = DataLoader(val_dataset, batch_size=batch_size)
        preds, labels = eval_model(val_dl, model, tokenizer)


    out_path = os.path.join(WEIGHTS_PATH, uuid)
    try: 
        print(f"dumping model weights to {out_path}...")
        torch.save(model.state_dict(), out_path)
        print("done.")
        return True
    except: 
        return False 

class SRDataset(Dataset):
    ''' A torch Dataset wrapper for screening corpora '''
    def __init__(self, titles, abstracts, labels): 
        super(SRDataset).__init__()
        self.titles = titles
        self.abstracts = abstracts
        self.labels = labels 
        self.N = len(self.titles)

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, index) -> tuple:
        ti = self.titles[index]
        abstract = self.abstracts[index]
        y = self.labels[index]

        X = ti + " [sep] " + abstract
        return (X, y)



