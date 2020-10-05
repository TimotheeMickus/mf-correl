import torch
import torch.nn as nn
import collections
import csv
import os
import datetime
import pickle

def get_random_embedding():
	return torch.randn(1, 512)

if __name__ == "__main__":
	import argparse
	p = argparse.ArgumentParser("Make random sentence embeddings with an untrained LSTM")
	
	p.add_argument("--input_dir", type=str, help="input directory", required=True)
	p.add_argument("--pickle", type=str, help="pickle filepath to save model", default="randlstm.pkl")

	args = p.parse_args()
	print(datetime.datetime.now(), "process start")
	
	# model
	lstm = nn.LSTM(512, 512, 1)
	for name, param in lstm.named_parameters():
		if 'bias' in name:
			nn.init.constant_(param, 0.0)
		elif 'weight' in name:
			nn.init.xavier_normal_(param)

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
				input_tensors = torch.cat([vocab[w] for w in line.strip().split()])
				h, _ = lstm(input_tensors.unsqueeze(1))
				embedding = h[-1].view(-1)
				writer.writerow([line.strip(), " ".join(map(str, embedding.numpy()))])

	print(datetime.datetime.now(), "saving LSTM")
	with open(args.pickle, "wb") as ostr:
		pickle.dump((vocab, lstm), ostr)
	
