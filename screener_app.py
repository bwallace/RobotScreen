import json 
import os 
# seems wrong, but for some reason manually invoking garbage collection
# is necessary to release memory after predictions (?)
import gc 
import numpy as np 

from flask import Flask, jsonify, request

import screening_model
from screening_model import SRDataset

import torch 
from transformers import RobertaForSequenceClassification, RobertaTokenizer, PretrainedConfig
from torch.utils.data import DataLoader

app = Flask(__name__)

# for now, assuming predictions on cpu.
device = "cpu"

@app.route('/')
def hello():
    return 'Welcome to RobotScreener ;)'

@app.route('/train/<uuid>', methods=['POST'])
def train(uuid: str):
    #studies = json.loads(request.json)['articles']
    labeled_data = json.loads(request.json)['labeled_data']
    
    titles, abstracts, labels = [], [], []

    for citation in labeled_data:
        titles.append(citation['ti'])
        abstracts.append(citation['abs'])
        labels.append(int(citation['label']))
    
    
    dataset = SRDataset(titles, abstracts, np.array(labels))

    success = screening_model.train_and_save(dataset, uuid, batch_size=8, epochs=1)
    return f"success training? {success}" 

@app.route('/predict/<uuid>', methods=['POST'])
def predict(uuid: str):
    studies = json.loads(request.json)['input_citations']
    
    titles, abstracts = [], []

    for citation in studies:
        titles.append(citation['ti'])
        abstracts.append(citation['abs'])
    
    
    dataset = SRDataset(titles, abstracts)

    # we just outright assume that we are using Roberta; this will break
    # if untrue. TODO probably want to add flexibility here.
    tokenizer = RobertaTokenizer.from_pretrained("allenai/biomed_roberta_base") 
    model     = RobertaForSequenceClassification.from_pretrained("allenai/biomed_roberta_base", 
                                                                 num_labels=2).to(device=device) 

    # note that we assume a *.pt extension for the pytorch stuff.
    weights_path = os.path.join("saved_model_weights", uuid+".pt")
    print(f"loading model weights from {weights_path}...")
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    
    dl = DataLoader(dataset, batch_size=8)
    preds, _ = screening_model.make_preds(dl, model, tokenizer, device=device)
    
    # oddly without this memory will not be released following the predictions
    gc.collect()
    return {"predictions": preds}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

