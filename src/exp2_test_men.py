import numpy as np
import scipy.stats
import gensim

def cdist(v_a, v_b):
	return 1 - (v_a.dot(v_b)) / (np.linalg.norm(v_a) * np.linalg.norm(v_b))

def load_MEN(filepath):
	with open(filepath) as istr:
		data = map(str.strip, istr)
		data = map(str.split, data)
		*data, targets = zip(*data)
		data = [*data, map(float, targets)]
		data = zip(*data)
		data = list(data)
	vocab = {w for line in data for w in line[:2]}
	return data, vocab

def load_w2v_subset(filepath, vocab, has_header=True, is_binary=False):
	if is_binary:
		model = gensim.models.KeyedVectors.load_word2vec_format(filepath, binary=True)
		return {w:model[w] if w in model else np.random.normal(0, 1, 300) for w in vocab}
	with open(filepath, "r") as istr:
		if has_header:
			_ = next(istr)
		embs = {
			w: np.array(list(map(float, v.split())))
			for l in istr
			for w, v in [l.strip().split(' ', 1)]
			if w in vocab
		}
	return embs

def load_fasttext_subset(filepath, vocab):
	model = gensim.models.fasttext.load_facebook_model(filepath)
	return {w:model[w] for w in vocab}

def get_l2_seq(embs, men_pairs):
	return np.array([np.linalg.norm(embs[a] - embs[b]) for a, b in men_pairs])

def get_cdist_seq(embs, men_pairs):
	return np.array([cdist(embs[a], embs[b]) for a, b in men_pairs])

if __name__ == "__main__":
	import argparse
	p = argparse.ArgumentParser("Compute results on MEN for one set of embeddings")
	p.add_argument("--embs", type=str, help="input embeddings to test", required=True)
	p.add_argument("--has_header", action="store_true")
	p.add_argument("--is_fasttext", action="store_true")
	p.add_argument("--is_binary", action="store_true")
	p.add_argument("--men_path", type=str, required=True)
	p.add_argument("--output_file", type=str, required=True)
	args = p.parse_args()

	data, vocab = load_MEN(args.men_path)

	embs = load_fasttext_subset(args.embs, vocab) if args.is_fasttext\
		else load_w2v_subset(args.embs, vocab, has_header=args.has_header, is_binary=args.is_binary)

	A, B, targets = zip(*data)
	men_pairs = list(zip(A, B))
	l2_seq = get_l2_seq(embs, men_pairs)
	cdist_seq = get_cdist_seq(embs, men_pairs)

	with open(args.output_file, "w") as ostr:
		print("l2: spearman:", *scipy.stats.spearmanr(l2_seq, targets), ", pearson:", *scipy.stats.pearsonr(l2_seq, targets), file=ostr)
		print("cdist: spearman:", *scipy.stats.spearmanr(cdist_seq, targets), ", pearson:", *scipy.stats.pearsonr(cdist_seq, targets), file=ostr)
