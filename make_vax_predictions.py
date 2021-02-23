import os 

import numpy as np 
import pandas as pd 

import json 

import screening_model
from screening_model import SRDataset

import torch 
from transformers import RobertaForSequenceClassification, RobertaTokenizer, PretrainedConfig
from torch.utils.data import DataLoader

device = "cuda"

def make_predictions_from_json(inputs_path="example-datasets/2021trials.json", uuid="vaccine_model") -> None:
    
    json_data = json.load(open("example-datasets/2021trials.json", 'r'))
    titles, abstracts = [], []
    
    for study in json_data:
        titles.append(study["ti"])
        abstracts.append(study["ab"])


    dataset = SRDataset(titles, abstracts)

    tokenizer = RobertaTokenizer.from_pretrained("allenai/biomed_roberta_base") 
    model     = RobertaForSequenceClassification.from_pretrained("allenai/biomed_roberta_base", 
                                                                 num_labels=2).to(device=device) 

    # note that we assume a *.pt extension for the pytorch stuff.
    weights_path = os.path.join("saved_model_weights", uuid+".pt")
    print(f"loading model weights from {weights_path}...")
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    
    dl = DataLoader(dataset, batch_size=8)
    preds, _ = screening_model.make_preds(dl, model, tokenizer, device=device)
    df = pd.DataFrame({"titles":titles, "abstracts":abstracts, "predictions":preds})
    df.to_csv("vaccine_preds.csv", index=False) 

make_predictions_from_json()
