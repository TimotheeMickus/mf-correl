from distances import levenshtein, levenshtein_normalised, l2, cdist, jaccard

import csv
import string
from nltk.corpus import wordnet as wn
import collections
import itertools
import numpy as np
import spacy
spacy.load('en_core_web_sm')
import json
import multiprocessing
import tqdm


def parse_line(line):
	sentence, emb = line
	sentence, emb = sentence.split(), emb.split()
	emb = np.array(list(map(float, emb)))
	return (list(sentence), emb)


def read_tsv(fn):
	with open(fn) as istr:
		data = csv.reader(istr, delimiter="\t")
		data = map(parse_line, data)
		data = enumerate(data)
		data = list(data)
	return data


def make_pairs(data):
	return itertools.combinations(data, 2)


def to_chr_seq(w2c, s):
	return ''.join(map(chr, map(w2c.__getitem__, s)))


def default_synonym(word):
	cands = wn.synsets(word)
	if not cands: return word
	return cands[0].lemma_names()[0]


def full_pair_to_json(pair,
	w2c=collections.defaultdict(itertools.count().__next__),
	control_synonyms=True, meanings_only=False,	include_jaccard=True,
	stops=spacy.lang.en.stop_words.STOP_WORDS | {'d', 's', "'", 're', 've', 'll', 'm'} | set(string.punctuation)):
	# unpack
	(i1, (s1, v1)), (i2, (s2, v2)) = pair

	# meaning scores
	meaning_scores = {"l2":l2(v1, v2), "cdist":cdist(v1, v2)}
	if meanings_only:
		jdict = {
			"idx": [i1, i2],
			#"sentences":[s1,s2], #useful for debugging, but eats up a lot of space.
			"meaning_scores":meaning_scores,
			"text_scores":{},
		}
		return json.dumps(jdict)

	# text scores: preproc
	# convert sentences to reps
	c1, c2 = to_chr_seq(w2c, s1), to_chr_seq(w2c, s2)
	# filter stop words
	s1_f, s2_f = (w for w in s1 if w not in stops), (w for w in s2 if w not in stops)
	c1_f, c2_f = to_chr_seq(w2c, s1_f), to_chr_seq(w2c, s2_f)

	# text scores
	text_scores = {
		"lev":levenshtein(c1, c2),
		"lev_n":levenshtein_normalised(c1, c2),
		"lev_f":levenshtein(c1_f, c2_f),
		"lev_fn":levenshtein_normalised(c1_f, c2_f),
	}
	if include_jaccard:
		text_scores.update({
			"jac":jaccard(c1, c2),
			"jac_f":jaccard(c1_f, c2_f),
		})
	if control_synonyms:
		syn_s1 = list(map(default_synonym, s1))
		syn_s2 = list(map(default_synonym, s2))
		syn_c1, syn_c2 = to_chr_seq(w2c, syn_s1), to_chr_seq(w2c, syn_s2)
		syn_s1_f, syn_s2_f = (w for w in syn_s1 if w not in stops), (w for w in syn_s2 if w not in stops)
		syn_c1_f, syn_c2_f = to_chr_seq(w2c, syn_s1_f), to_chr_seq(w2c, syn_s2_f)
		text_scores.update({
			"lev_syn":levenshtein(syn_c1, syn_c2),
			"lev_syn_n":levenshtein_normalised(syn_c1, syn_c2),
			"lev_syn_f":levenshtein(syn_c1_f, syn_c2_f),
			"lev_syn_fn":levenshtein_normalised(syn_c1_f, syn_c2_f),
		})
		if include_jaccard:
			text_scores.update({
				"jac_syn":jaccard(syn_c1, syn_c2),
				"jac_syn_f":jaccard(syn_c1_f, syn_c2_f),
			})

	# return
	jdict = {
		"idx": [i1, i2],
		#"sentences":[s1,s2], #useful for debugging, but eats up a lot of space.
		"meaning_scores":meaning_scores,
		"text_scores":text_scores,
	}
	return json.dumps(jdict)


# specific pairwise distance functions
def meaning_only(pair):
	return full_pair_to_json(pair, meanings_only=True)


def no_jaccard(pair):
	return full_pair_to_json(pair, include_jaccard=False)


def no_synonyms(pair):
	return full_pair_to_json(pair, control_synonyms=False)


def no_synonyms_no_jaccard(pair):
	return full_pair_to_json(pair, control_synonyms=False, include_jaccard=False)


if __name__=="__main__":
	import argparse

	p = argparse.ArgumentParser("""Computing distances for sentences pairs.
		Takes as input sentence embedding + tokenized sentence TSV.
		Produces one JSON per sentence pair.""")
	p.add_argument("--input", type=str, help="input file", required=True)
	p.add_argument("--output", type=str, help="output file", default="output.json")
	p.add_argument("--meaning_only", action="store_true", help="only meaning distances")
	p.add_argument("--no_jaccard", action="store_true", help="don't compute jaccard distances")
	p.add_argument("--no_synonyms", action="store_true", help="don't control for synonymy")

	args = p.parse_args()

	sample = make_pairs(read_tsv(args.input))

	pairing_func = None
	if args.meaning_only:
		pairing_func = meaning_only
	elif args.no_jaccard:
		if args.no_synonyms:
			pairing_func = no_synonyms_no_jaccard
		else:
			pairing_func = no_jaccard
	elif args.no_synonyms:
		pairing_func = no_synonyms
	else:
		pairing_func = full_pair_to_json

	with open(args.output, "w") as ostr, multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
		for jstring in tqdm.tqdm(pool.imap_unordered(pairing_func, sample, 200), total=(4123 * 4122) // 2):
			print(jstring, file=ostr)
	pool.close()
	pool.join()
