import urllib.request
from Bio import SeqIO
import pandas as pd
import gzip
import os
from collections import defaultdict

# Function to download the PDB sequences file
def download_pdb_sequences(filename='pdb_seqres.txt.gz'):
    url = 'ftp://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz'
    print('Downloading PDB sequences...')
    urllib.request.urlretrieve(url, filename)
    print('Download complete.')

# Function to parse the PDB sequences and extract single subunit proteins
def parse_single_chain_proteins(filename='pdb_seqres.txt.gz'):
    # Store the PDB IDs and their associated chains
    pdb_chain_map = defaultdict(set)
    protein_data = defaultdict(list)
    
    with gzip.open(filename, 'rt') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            pdb_id = record.id[0:4].upper()
            chain_id = record.id[5:]
            description = record.description
            sequence = str(record.seq)
            
            # Only include protein sequences
            if 'mol:protein' in description.lower():
                pdb_chain_map[pdb_id].add(chain_id)
                protein_data[pdb_id].append({
                    'PDB ID': pdb_id,
                    'Protein Name': description.split(None, 4)[-1],
                    'Length': len(sequence),
                    'Amino Acid Sequence': sequence
                })

    # Filter for PDB entries that have only a single chain
    single_chain_proteins = []
    for pdb_id, chains in pdb_chain_map.items():
        if len(chains) == 1:
            # There is only one chain for this PDB ID, so we keep the entry
            single_chain_proteins.extend(protein_data[pdb_id])
    
    print(f"Found {len(single_chain_proteins)} single-chain proteins.")
    return single_chain_proteins

# Function to save data to Excel
def save_to_excel(data, output_filename='single_chain_protein_sequences.xlsx'):
    df = pd.DataFrame(data)
    df.to_excel(output_filename, index=False)
    print(f'Data saved to {output_filename}')

# Main function to execute the workflow
def main():
    pdb_seq_filename = 'pdb_seqres.txt.gz'
    if not os.path.exists(pdb_seq_filename):
        download_pdb_sequences(pdb_seq_filename)
    data = parse_single_chain_proteins(pdb_seq_filename)
    save_to_excel(data)

if __name__ == '__main__':
    main()