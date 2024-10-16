import urllib.request
from Bio import SeqIO
import pandas as pd
import gzip
import os

# Function to download the PDB sequences file
def download_pdb_sequences(filename='pdb_seqres.txt.gz'):
    url = 'ftp://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz'
    print('Downloading PDB sequences...')
    urllib.request.urlretrieve(url, filename)
    print('Download complete.')

# Function to parse the PDB sequences and extract protein sequences
def parse_pdb_sequences(filename='pdb_seqres.txt.gz'):
    data = []
    with gzip.open(filename, 'rt') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            description = record.description
            seq = str(record.seq)
            pdb_id = record.id[0:4].upper()
            chain_id = record.id[5:]
            # Filter for proteins
            if 'mol:protein' in description.lower():
                # Extract protein name from description
                # Example description: '>1ABC:A mol:protein length:130  Myoglobin'
                parts = description.split()
                # Find the index of 'length:'
                try:
                    length_index = parts.index(next(s for s in parts if s.startswith('length:')))
                    # Protein name is everything after the length information
                    protein_name = ' '.join(parts[length_index + 1:])
                except StopIteration:
                    protein_name = ' '.join(parts[4:])  # Fallback if 'length:' is not found
                data.append({
                    'PDB ID': pdb_id,
                    'Chain ID': chain_id,
                    'Protein Name': protein_name,
                    'Length':len(seq),
                    'Amino Acid Sequence': seq
                })
    return data

# Function to save data to Excel
def save_to_excel(data, output_filename='all_PDB_protein_sequences.xlsx'):
    df = pd.DataFrame(data)
    df.to_excel(output_filename, index=False)
    print(f'Data saved to {output_filename}')

# Main function to execute the workflow
def main():
    pdb_seq_filename = 'pdb_seqres.txt.gz'
    if not os.path.exists(pdb_seq_filename):
        download_pdb_sequences(pdb_seq_filename)
    data = parse_pdb_sequences(pdb_seq_filename)
    save_to_excel(data)

if __name__ == '__main__':
    main()