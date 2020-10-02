import torch
import torch.nn as nn
import collections
import csv
import os
import datetime
import pickle
import math

def get_random_embedding():
	return torch.randn(1, 512)

if __name__ == "__main__":
	import argparse
	p = argparse.ArgumentParser("Make random sentence embeddings with an untrained Transformer")
	
	p.add_argument("--input_dir", type=str, help="input directory", required=True)
	p.add_argument("--pickle", type=str, help="pickle filepath to save model", default="randlstm.pkl")

	args = p.parse_args()
	print(datetime.datetime.now(), "process start")
	
	# model
	encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
	transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
	for name, param in transformer.named_parameters():
		nn.init.uniform_(param, -0.1, 0.1)
		if param.dim() > 1:
			nn.init.xavier_uniform_(param)
		

	pe = torch.zeros(512, 512)
	position = torch.arange(0, 512, dtype=torch.float).unsqueeze(1)
	div_term = torch.exp(torch.arange(0, 512, 2).float() * (-math.log(10000.0) / 512))
	pe[:, 0::2] = torch.sin(position * div_term)
	pe[:, 1::2] = torch.cos(position * div_term)
	pe = pe.unsqueeze(0).transpose(0, 1)

	# data
	input_files = (
		os.path.join(root, filename)
		for root, _, filenames in os.walk(args.input_dir) 
		for filename in filenames 
		if filename.endswith(".txt")
	)
	vocab = collections.defaultdict(get_random_embedding)

	torch.set_grad_enabled(False)
	for input_file in input_files:	
		print(datetime.datetime.now(), "handling file %s" % input_file)
		with open(input_file) as istr, open(input_file+ ".rd.tsv", "w") as ostr:
			writer = csv.writer(ostr, delimiter="\t")
			for line in istr:
				ipt = torch.cat([vocab[w] for w in line.strip().split()])
				h = transformer(ipt.unsqueeze(1) + pe[:ipt.size(0),:])
				embedding = h.sum(0).view(-1) / math.sqrt(ipt.size(0))
				writer.writerow([line.strip(), " ".join(map(str, embedding.numpy()))])

	print(datetime.datetime.now(), "saving Transformer")
	with open(args.pickle, "wb") as ostr:
		pickle.dump((vocab, transformer), ostr)
	