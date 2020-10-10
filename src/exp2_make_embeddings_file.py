from datetime import datetime
import gensim
import numpy as np
import csv

def load_embs_subset(filepath, vocab, is_fasttext, has_header, is_binary):
    if is_fasttext:
        return load_fasttext_subset(filepath, vocab)
    return load_w2v_subset(filepath, vocab, has_header=has_header, is_binary=is_binary)

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

def to_tsv(data, output_file, embs):
    with open(output_file, "w") as ostr:
        writer = csv.writer(ostr, delimiter="\t")
        for word, def in data:
            emb_str = " ".join(map(str, embs[word].tolist()))
            _ = writer.writerow([def, emb_str])

def from_csv(input_file):
    with open(input_file) as istr:
        reader = csv.reader(istr)
        data = list(reader)
    return data

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser("Transform data CSV to emb,def TSV.")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--embs_path", type=str, required=True)
    parser.add_argument("--is_fasttext", action="store_true")
    parser.add_argument("--has_header", action="store_true")
    parser.add_argument("--is_binary", action="store_true")
    args = parser.parse_args()

    print(datetime.now(), "converting %s to %s using %s" % (args.input_file, args.output_file, args.embs_path))

    data = from_csv(args.input_file)
    vocab, defs = zip(*data)
    embs = load_embs_subset(args.embs_path, vocab, args.is_fasttext, args.has_header, args.is_binary)
    to_tsv(data, args.output_file, embs)
