import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import BertForMaskedLM, BertTokenizer, pipeline, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

def Tokenize_Input(excl):
	tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
	
	df = pd.read_excel(excl)
	
	in_list = df["Amino Acid Sequence"].to_list()
	in_list = [" ".join(s) for s in in_list]

	tokenized_in = [tokenizer(in_list[i], return_tensors = 'pt', padding = False)["input_ids"].flatten() for i in range(len(in_list))]
	
	torch.save(pad_sequence(tokenized_in, batch_first=True), excl.replace(".xlsx", ".pt"))


Tokenize_Input("single_chain_protein_sequences.xlsx")