'''
RobotScreener

Simple module for consuming labeled abstract screening data, training
a model on the basis of this, and dumping it to disk.
'''
import os 
from typing import Type, Tuple
import copy 
import collections 

import numpy as np 

import sklearn # just for evaluation 
from sklearn import metrics 

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader, WeightedRandomSampler
from torch import optim

import transformers
from transformers import AdamW
from transformers import RobertaForSequenceClassification, RobertaTokenizer, PretrainedConfig

###
# Globals; need to be edited for server.
import config

WEIGHTS_PATH = config.weights_path_str
###

def train(dl: DataLoader, epochs: int = 3, val_dataset: Dataset = None, recall_weight: int = 10) -> Tuple[collections.OrderedDict, Type[transformers.PreTrainedTokenizer]]:
    ''' 
    Trains and returns a model over the data in dl. If a validation Dataset is provided,
    the model will be evaluated on this set per epoch, and the model w/the best performance
    will be returned; note that 'best' here depends on the `recall_weight', which dictates
    how much recall (to class `1', assumed to be includes) is weighted relative to precision.
    '''
    
    ''' Model and optimizer ''' 
    tokenizer = RobertaTokenizer.from_pretrained("allenai/biomed_roberta_base") 
    model     = RobertaForSequenceClassification.from_pretrained("allenai/biomed_roberta_base", 
                                                                 num_labels=2).to(device=config.device_str) 
    #for param in list(model.parameters())[-1:]:
    #    param.requires_grad = False
    
    #optimizer = AdamW(model.parameters())
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    best_val = -np.inf
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
            model_outputs = model(torch.tensor(batch_X_tensor['input_ids']).to(device=config.device_str), 
                              attention_mask=torch.tensor(batch_X_tensor['attention_mask']).to(device=config.device_str), 
                              labels=batch_y_tensor.to(device=config.device_str))
          
            model_outputs['loss'].backward()
           
            running_losses.append(model_outputs['loss'].detach().float())
            if batch_num % 10 == 0:
                avg_loss = sum(running_losses[-10:])/len(running_losses[-10:])
                print(f"avg loss for last 10 batches: {avg_loss}")
            optimizer.step()

        if val_dataset is not None: 
            # note that we use the same batchsize for val as for train
            val_dl = DataLoader(val_dataset, batch_size=dl.batch_size)
            preds, labels = make_preds(val_dl, model, tokenizer, device=config.device_str)
            results = classification_eval(preds, labels, threshold=0.5)
            # composite score; ad-hoc, I know
            score = recall_weight*results['recall'][1] + results['precision'][1]
            results["score"] = score
            print(results)

            if score > best_val:
                print("found new best parameter set; saving.")
                best_model_state = copy.deepcopy(model.state_dict())
                best_val = score
        else:
           best_model_state = model.state_dict()

    return best_model_state, tokenizer
    
def make_preds(val_data: DataLoader, model: Type[torch.nn.Module], tokenizer: Type[transformers.PreTrainedTokenizer], device: str="cuda") -> Tuple:
    preds, labels = [], []
    with torch.no_grad():
        model.eval()
        for (X, y) in val_data:
            
            batch_X_tensor = tokenizer.batch_encode_plus(X, max_length=512, 
                                                        add_special_tokens=True, 
                                                        pad_to_max_length=True)
            model_outputs = model(torch.tensor(batch_X_tensor['input_ids']).to(device=device), 
                              attention_mask=torch.tensor(batch_X_tensor['attention_mask']).to(device=device))
            
            probs = torch.softmax(model_outputs['logits'].cpu(), 1)[:,1]
            preds.extend(probs.tolist())
            labels.extend(y.tolist())
       
    return (preds, labels)

def classification_eval(preds: list, labels: list, threshold: float = 0.5) -> dict:
    y_preds = np.array(preds)
    y_preds_binary = np.where(y_preds > threshold, 1, 0)
    (p, r, f, s) = metrics.precision_recall_fscore_support(labels, y_preds_binary)
    auc = sklearn.metrics.roc_auc_score(labels, y_preds)
    print (metrics.classification_report(labels, y_preds_binary))
    print(f"auc: {auc}")
    return {"precision":p, "recall":r, "f":f, "AUC":auc}

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
                    epochs: int = 5, val_dataset: Dataset = None) -> bool:
    '''
    Trains a classification model on the given review dataset and dumps
    to disk. If a val_dataset is provided, performance is evaluated on 
    this each epoch, and the best model is saved.
    '''

    # this is a sampler that assigns larger sampling weights to (rare) positive
    # examples for batch construction, to account for data imbalance.
    weighted_sampler = get_weighted_sampler(sr_dataset)
    
    dl = DataLoader(sr_dataset, batch_size=batch_size, sampler=weighted_sampler)
    model_state, tokenizer = train(dl, epochs=epochs, val_dataset=val_dataset)

    out_path = os.path.join(WEIGHTS_PATH, uuid+".pt")
    try: 
        print(f"dumping model weights to {out_path}...")
        torch.save(model_state, out_path)
        print("done.")
        return True
    except: 
        return False 

class SRDataset(Dataset):
    ''' A torch Dataset wrapper for screening corpora '''
    def __init__(self, titles, abstracts, labels=None): 
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

        X = ti + " [sep] " + abstract

        if self.labels is not None: 
            y = self.labels[index]
            return (X, y)

        return (X, 0)



