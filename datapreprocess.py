import pandas as pd
from Bio import SeqIO
from itertools import product
from tqdm import tqdm
import numpy as np

def generate_kmer_vector(sequence, k=6):
    kmer_counts = {"".join(kmer): 0 for kmer in product('ACGT', repeat=k)}
    
    # Count k-mers in the sequence
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k].upper()
        if kmer in kmer_counts:
            kmer_counts[kmer] += 1
    
    kmer_vector = np.array(list(kmer_counts.values()))
    kmer_vector = np.exp(kmer_vector) / np.sum(np.exp(kmer_vector))
    
    return kmer_vector.tolist()

def process_data(csv_file, fasta_file, output_file):
    df = pd.read_csv(csv_file)
    
    sequences = {record.id: str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")}
    
    output_data = []
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows"):
        seq_id = row['seqname']
        start, end = row['start'], row['end']
        feature = row['feature']
        
        if seq_id in sequences:
            sequence = sequences[seq_id][start-1:end]  # -1 because Python indexing starts at 0
            kmer_vector = generate_kmer_vector(sequence)
            output_data.append({'Vector': kmer_vector, 'Feature': feature})
    
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_file, index=False)
    print(f"Processed data has been saved to {output_file}")

if __name__ == "__main__":
    csv_file = 'balanced_arab_data.csv'  
    fasta_file = 'arabidopsis_genome.fasta'  
    output_file = 'processed_kmer_vectors_softmax.csv'  
    process_data(csv_file, fasta_file, output_file)