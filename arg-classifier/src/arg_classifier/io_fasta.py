from Bio import SeqIO
from pathlib import Path

def load_fasta(fasta_path):
    """Load sequences from FASTA file."""
    sequences = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append({
            'id': record.id,
            'sequence': str(record.seq).upper()
        })
    return sequences

def save_fasta(sequences, output_path):
    """Save sequences to FASTA file."""
    with open(output_path, 'w') as f:
        for seq in sequences:
            f.write(f">{seq['id']}\n{seq['sequence']}\n")
