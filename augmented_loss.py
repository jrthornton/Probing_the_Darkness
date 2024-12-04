from Bio.Blast import NCBIWWW, NCBIXML
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from transformers import BertForMaskedLM, BertTokenizer, pipeline, AutoTokenizer

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

def blast_sequences_against_pdb(query_sequences, output_file="blast_results.xml"):
    records = [SeqRecord(Seq(seq), id=f"query_{i}") for i, seq in enumerate(query_sequences)]
    fasta_str = "\n".join(record.format("fasta") for record in records)
    
    print("Running BLAST...")
    result_handle = NCBIWWW.qblast("blastp", "pdb", fasta_str)
    with open(output_file, "w") as out_file:
        out_file.write(result_handle.read())

# Function to parse BLAST results
def parse_blast_results(xml_file):
    with open(xml_file) as result_handle:
        blast_records = NCBIXML.parse(result_handle)
        results = {}
        for blast_record in blast_records:
            query_id = blast_record.query_id
            if blast_record.alignments:
                top_hit = blast_record.alignments[0]
                hsp = top_hit.hsps[0]
                results[query_id] = {
                    "pdb_id": top_hit.hit_id.split("|")[0],
                    "e_value": hsp.expect,
                    "score": hsp.score,
                    "alignment_length": hsp.align_length,
                }
            else:
                results[query_id] = None
    return results

def aug_loss(config, gen, lengths):
    # Run BLAST searches for the generated sequences
    sequences = []
    for t in range(config['generator']['sim_num']):
        seq = gen[t, :].tolist()
        seq = tokenizer.convert_ids_to_tokens(seq)
        sequences.append(''.join(seq[:lengths[t]]))
    
    blast_sequences_against_pdb(sequences, output_file="blast_results.xml")
    
    # Parse the BLAST results
    blast_results = parse_blast_results("blast_results.xml")
    
    # Get e-value threshold from config or set default
    e_value_threshold = config['training']['e_value_threshold']  # Adjust threshold as needed
    
    # Count the number of sequences with significant hits
    significant_count = 0
    total_sequences = config['generator']['sim_num']
    
    for query_id, result in blast_results.items():
        if result and result['e_value'] <= e_value_threshold:
            significant_count += 1
        else:
            print(f"{query_id}: No significant hits or e-value above threshold.")
    
    # Compute the fraction of significant sequences
    fraction_significant = significant_count / total_sequences
    return fraction_significant

