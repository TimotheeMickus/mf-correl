from __future__ import print_function

import numpy as np
import scipy.stats
 #import torch

def cdist(v_a, v_b):
	return 1 - (v_a.dot(v_b)) / (np.linalg.norm(v_a) * np.linalg.norm(v_b))

def load_SICK(filepath):
	with open(filepath) as istr:
		_ = next(istr)
		data = map(str.strip, istr)
		data = (l.split("\t") for l in data)
		data = ([l[1], l[2], float(l[4])] for l in data)
		data = list(data)
	vocab = {sent for line in data for sent in line[:2]}
	return data, vocab

def get_infersent_embs(filepath, vocab):
	from exp3_embs.InferSent.models import InferSent
	model_version = 1
	MODEL_PATH = "InferSent/encoder/infersent%s.pkl" % model_version
	params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
		        'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
	model = InferSent(params_model)
	model.load_state_dict(torch.load(MODEL_PATH))
	W2V_PATH = 'embs/glove-840B/glove.840B.300d.txt'
	model.set_w2v_path(W2V_PATH)
	model.build_vocab_k_words(K=100000)
	vocab = list(vocab)
	embs = dict(zip(vocab, model.encode(vocab, bsize=128, tokenize=False, verbose=True)))
	return embs


def get_USE_DAN_embs(filepath, vocab):
	module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
	return _get_USE_embs(filepath, vocab, module_url)


def get_USE_Tf_embs(filepath, vocab):
	module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
	return _get_USE_embs(filepath, vocab, module_url)


def _get_USE_embs(filepath, vocab, module_url):
	import tensorflow as tf
	import tensorflow_hub as hub

	vocab = list(vocab)
	with tf.Session() as session:
		model = hub.load(module_url)
		init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
		session.run(init_op)
		message_embeddings = model(vocab).eval()
		embs = dict(zip(vocab, np.array(message_embeddings)))
	return embs

def get_skipthought_embs(filepath, vocab):
	from exp3_embs.skip_thoughts import skipthoughts
	model = skipthoughts.load_model()
	encoder = skipthoughts.Encoder(model)
	vocab = list(vocab)
	vectors = encoder.encode(vocab)
	embs = dict(zip(vocab, np.array(vectors)))
	return embs

# fix for pickle
def get_random_embedding():
	return torch.randn(1, 512)

def get_randlstm_emb(filepath, vocab):
	from exp3_embs.get_randlstm_embs import get_random_embedding
	import pickle
	import torch
	with open(filepath, "rb") as istr:
		embedding_maker, lstm = pickle.load(istr)
	vocab = list(vocab)
	embs = {}
	with torch.no_grad():
		for sentence in vocab:
			input_tensors = torch.cat([embedding_maker[w] for w in sentence.strip().split()])
			h, _ = lstm(input_tensors.unsqueeze(1))
			embedding = h[-1].view(-1).numpy()
			embs[sentence] = embedding
	return embs

def get_randtf_emb(filepath, vocab):
	from exp3_embs.get_randtf_embs import get_random_embedding
	import pickle
	import math
	import torch
	with open(filepath, "rb") as istr:
		embedding_maker, transformer = pickle.load(istr)
	vocab = list(vocab)

	pe = torch.zeros(512, 512)
	position = torch.arange(0, 512, dtype=torch.float).unsqueeze(1)
	div_term = torch.exp(torch.arange(0, 512, 2).float() * (-math.log(10000.0) / 512))
	pe[:, 0::2] = torch.sin(position * div_term)
	pe[:, 1::2] = torch.cos(position * div_term)
	pe = pe.unsqueeze(0).transpose(0, 1)

	embs = {}
	with torch.no_grad():
		for sentence in vocab:
			ipt = torch.cat([embedding_maker[w] for w in sentence.strip().split()])
			h = transformer(ipt.unsqueeze(1) + pe[:ipt.size(0),:])
			embedding = h.sum(0).view(-1) / math.sqrt(ipt.size(0))
			embs[sentence] = embedding
	return embs

def get_rand_emb(filepath, vocab):
	from exp3_embs.get_rand_embs import get_random_embedding
	import pickle
	with open(filepath, "rb") as istr:
		embedding_maker = pickle.load(istr)
	embs = {sentence:embedding_maker[sentence].view(-1) for sentence in vocab}
	return embs

def get_l2_seq(embs, sick_pairs):
	return np.array([np.linalg.norm(embs[a] - embs[b]) for a, b in sick_pairs])

def get_cdist_seq(embs, sick_pairs):
	return np.array([cdist(embs[a], embs[b]) for a, b in sick_pairs])

if __name__ == "__main__":
	import argparse
	p = argparse.ArgumentParser("Compute results on SICK for one set of embeddings")
	p.add_argument("--emb_path", type=str, help="input embeddings to test", required=False)
	p.add_argument("--emb_arch", type=str, help="input embeddings architecture", required=True,
		choices=["infersent", "DAN", "USE", "skipthoughts", "randlstm", "randtf", "randembs"])
	p.add_argument("--sick_path", type=str, help="path to SICK.txt", required=True)

	args = p.parse_args()

	data, vocab = load_SICK(args.sick_path)

	load_func = {
		"infersent":get_infersent_embs,
		"DAN":get_USE_DAN_embs,
		"USE":get_USE_Tf_embs,
		"skipthoughts":get_skipthought_embs,
		"randlstm":get_randlstm_emb,
		"randtf":get_randtf_emb,
		"randembs":get_rand_emb,
	}[args.emb_arch]

	embs = load_func(args.emb_path, vocab)

	A, B, targets = zip(*data)
	sick_pairs = list(zip(A, B))
	l2_seq = get_l2_seq(embs, sick_pairs)
	cdist_seq = get_cdist_seq(embs, sick_pairs)
	sr_l2, sp_l2 = scipy.stats.spearmanr(l2_seq, targets)
	pr_l2, pp_l2 = scipy.stats.pearsonr(l2_seq, targets)
	sr_cd, sp_cd = scipy.stats.spearmanr(cdist_seq, targets)
	pr_cd, pp_cd = scipy.stats.pearsonr(cdist_seq, targets)
	print("l2: spearman:", sr_l2, sp_l2, ", pearson:", pr_l2, pp_l2)
	print("cdist: spearman:", sr_cd, sp_cd, ", pearson:", pr_cd, pp_cd)
