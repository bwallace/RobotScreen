import json 

import numpy as np 

from flask import Flask, jsonify, request

import screening_model
from screening_model import SRDataset

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello World!'

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

    success = screening_model.train_and_save(dataset, uuid, batch_size=2)
    return f"success training? {success}" 