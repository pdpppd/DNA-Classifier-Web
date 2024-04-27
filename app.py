from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn as nn
import numpy as np
from inference import predict_and_get_lists, load_model
from itertools import product

app = Flask(__name__)

def generate_kmer_vector(sequence, k=6):
    # Initialize an empty dictionary for k-mer counts
    kmer_counts = {"".join(kmer): 0 for kmer in product('ACGT', repeat=k)}
    
    # Count k-mers in the sequence
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k].upper()
        if kmer in kmer_counts:
            kmer_counts[kmer] += 1
    
    # Convert the counts to a vector and apply softmax normalization
    kmer_vector = np.array(list(kmer_counts.values()))
    kmer_vector = np.exp(kmer_vector) / np.sum(np.exp(kmer_vector))
    
    return kmer_vector.tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    dna_sequence = request.form['dna_sequence']
    vector = generate_kmer_vector(dna_sequence)
    model, label_encoder = load_model('model.pth')
    data, labels = predict_and_get_lists(model, vector, label_encoder)
    return jsonify({'labels': labels, 'data': data})

if __name__ == '__main__':
    app.run(debug=True)