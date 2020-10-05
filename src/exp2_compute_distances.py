import random
import re
import itertools
import functools
import subprocess
import collections

import numpy as np
import scipy.stats
import Levenshtein
import torch
import multiprocessing

from allennlp.predictors.predictor import Predictor

import argparse
import sys
import json
import spacy
import tqdm

# TODO: modularize distances

#### Meaning distances
def cdist(v1, v2):
	return 1.0 - (np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def l2(v1, v2) :
	return np.linalg.norm(v1 - v2)


#### Text distances
@functools.lru_cache(maxsize=524288)
def apted(tree1, tree2):
	"""
		Call to apted JAR.
		This shouldn't be called, as it's grossly inefficient.
	"""
	cmd = 'java -jar apted.jar -t %s %s' % (tree1, tree2)
	apted_output = subprocess.check_output(cmd.split())
	score = float(apted_output.decode("utf-8"))
	return score

@functools.lru_cache(maxsize=524288)
def jaccard(seq1, seq2):
	"""
		Compute jaccard index
	"""
    union = len(seq1)
    intersection = 0
    d = collections.defaultdict(int)
    for i in seq1:
        d[i] += 1
    for i in seq2:
        x = d[i]
        if(x > 0):
            d[i] -= 1
            intersection += 1
        else:
            union += 1
    if not union: sys.stderr.write('errored on sequences: %s %s\n' % (str(seq1), str(seq2)) )
    return (1 - (intersection / union)) if union else 0.

@functools.lru_cache(maxsize=524288)
def levenshtein(str1, str2, normalise=False):
	"""
		Compute Levenshtein distance.
		Optionally normalize.
	"""
	tmp = Levenshtein.distance(str1, str2)
	if(normalise) and (len(str1) + len(str2)): tmp /= max(len(str1), len(str2))
	return tmp

@functools.lru_cache(maxsize=524288)
def levenshtein_normalised(str1, str2):
	"""
		Shorthand to compute Levenshtein distance + length normalization
	"""
    return levenshtein(str1, str2, normalise=True)


#### Misc functions
@functools.lru_cache(maxsize=524288)
def to_brkt(tree):
	"""
		Convert AllenNLP square bracket tree format to apted curly bracket tree format
	"""
	prep = tree.replace('(', '{').replace(')', '}')
	prep = re.sub(r" ([^{} ]+)}",r" {\1}}", prep)
	return prep.replace(' ', '')


def word_embedding_levenshtein(seq1, seq2, embeddings, average_distance, r=0.9, normalise=False):
	"""
		Compute Levenshtein w/ embedding similarity-based weigthing
	"""
    x1 = 1 + len(seq1)
    x2 = 1 + len(seq2)

    alpha = r / ((1 - r) * average_distance)

    # Initialisation of the matrix
    d = [] # Using Numpy structures for this is probably not more efficient
    d.append(list(range(x2)))
    for i in range(1, x1):
        d.append([i] * x2)

    # Core of the algorithm
    for i in range(1, x1):
        for j in range(1, x2):
            e1 = seq1[i-1]
            e2 = seq2[j-1]

            if(e1 == e2): c = 0
            else:
                v1 = embeddings[e1]
                v2 = embeddings[e2]

                if((v1 is None) or (v2 is None)): c = 1
                else:
                    dst = np.linalg.norm(v1 - v2) # Distance 2 (or L2 norm of the difference)

                    # Now, we need a function increasing function mapping 0 to 0 and +inf to 1
                    c = 1 - (1 / (1 + (alpha * dst)))

                    #c /= r # If you uncomment this line, the cost of a substitution at distance `average_distance` will be 1 and substitutions might have higher cost, up to 1/r. This might be justified as long as `r` is above 0.5 (otherwise, some substitutions might be more expensive than an insertion followed by a deletion).

            d[i][j] = min(
                (d[(i-1)][j] + 1), # Deletion of seq1[i]
                (d[i][(j-1)] + 1), # Insertion of seq2[j]
                (d[(i-1)][(j-1)] + c) # Substitution from seq1[i] to seq2[j]
            )

    raw = d[-1][-1]

    if(normalise): return (raw / (len(seq1) + len(seq2)))
    return raw



# `weights` is a dictionary of word to weight (between 0 and 1)
# One possibility for computing these weights: for a given token, let f be its document frequency (the proportion of documents in which it appears), then w = 1 - f
def weighted_levenshtein(seq1, seq2, weights, r=0.9, normalise=False):
	"""
		Compute Levenshtein w/ weighting
	"""
    x1 = 1 + len(seq1)
    x2 = 1 + len(seq2)

    alpha = r / ((1 - r) * average_distance)

    # Initialisation of the matrix
    d = [] # Using Numpy structures for this is probably not more efficient
    tmp = 0.0
    first_line = [tmp]
    for e in seq2:
        tmp += weights.get(e, 1)
        first_line.append(tmp)
    d.append(first_line)
    tmp = 0
    for e in seq1:
        tmp += weights.get(e, 1)
        d.append([tmp] * x2)

    # Core of the algorithm
    for i in range(1, x1):
        for j in range(1, x2):
            e1 = seq1[i-1]
            e2 = seq2[j-1]

            w1 = weights.get(e1, 1)
            w2 = weights.get(e2, 1)

            d[i][j] = min(
                (d[(i-1)][j] + w1), # Deletion of seq1[i]
                (d[i][(j-1)] + w2), # Insertion of seq2[j]
                (d[(i-1)][(j-1)] + (int(e1 != e2) * max(w1, w2))) # Substitution from seq1[i] to seq2[j]
			)

    raw = d[-1][-1]

    if(normalise): return (raw / (d[0][-1] + d[-1][0]))
    return raw


def sample_pairs(sentences, restrict_dataset_size=None, sample_size=1024, prevout2="../pairs-fn", prevout1="okays"):
	"""
		Get all sentences pairs.
	"""
	#sample = None
	#if False and prevout1 and prevout2:
	#	"""with open(prevout1) as istr:
	#		sample = map(str.strip, istr)
	#		sample = list(map(int, sample))
	#		sample = sorted(sample)"""

	#	with open(prevout2) as istr:
	#		lines = map(str.strip, istr)
	#		lines = [tuple(sorted([int(i) for i in l.split()])) for l in lines]
	#		lines = set(lines)
	sample = range(len(sentences))

	#if True or restrict_dataset_size:
	#	sample = random.sample(sample, 4096)

	sample = [p for p in itertools.combinations(sample, 2)]# if not tuple(sorted(p)) in lines]
	sys.stderr.write(str(len(sample)) + '\n')
	#sample = random.sample(sample, sample_size)
	return sample


def compute_val(single_arg):
	"""
		Compute all distances for a pair
	"""
	sentences, tree_idx, pos_decored_idx, meanings_idx, filtered_levenshtein, filtered_levenshtein_pos, w2c, remap, dump_vals, embeddings, average_distance, min_df, idx1, idx2 = single_arg
	if remap:
		meaning_1 = meanings_idx[mapping[idx1]]
		meaning_2 = meanings_idx[mapping[idx2]]
	else:
		meaning_1 = meanings_idx[idx1]
		meaning_2 = meanings_idx[idx2]

	cdist_score = cdist(meaning_1, meaning_2)
	l2_score = l2(meaning_1, meaning_2)

	# 1. get representations

	# 1a. sentence w/ misc. info added

	#tree1 = tree_idx[idx1]
	#tree2 = tree_idx[idx2]

	sentence_1 = sentences[idx1]
	sentence_2 = sentences[idx2]
	sentence_pos_1 = pos_decored_idx[idx1]
	sentence_pos_2 = pos_decored_idx[idx2]
	filtered_sentence_1 = filtered_levenshtein[idx1]
	filtered_sentence_2 = filtered_levenshtein[idx2]
	filtered_sentence_pos_1 = filtered_levenshtein_pos[idx1]
	filtered_sentence_pos_2 = filtered_levenshtein_pos[idx2]

	# 1b. character sequences for Levenshtein

	chars_1 = ''.join(chr(w2c[w]) for w in sentence_1)
	chars_2 = ''.join(chr(w2c[w]) for w in sentence_2)
	chars_pos_1 = ''.join(chr(w2c[w]) for w in sentence_pos_1)
	chars_pos_2 = ''.join(chr(w2c[w]) for w in sentence_pos_2)
	chars_f_1 = ''.join(chr(w2c[w]) for w in filtered_sentence_1)
	chars_f_2 = ''.join(chr(w2c[w]) for w in filtered_sentence_2)
	chars_fpos_1  = ''.join(chr(w2c[w]) for w in filtered_sentence_pos_1)
	chars_fpos_2  = ''.join(chr(w2c[w]) for w in filtered_sentence_pos_2)

	# 2. compute distances
	levenshtein_score = levenshtein(chars_1, chars_2)
	levenshtein_pos_score = levenshtein(chars_pos_1, chars_pos_2)
	levenshtein_n_score = levenshtein_normalised(chars_1, chars_2)
	levenshtein_f_score = levenshtein(chars_f_1, chars_f_2)
	levenshtein_fn_score = levenshtein_normalised(chars_f_1, chars_f_2)
	levenshtein_fpos_score = levenshtein(chars_fpos_1, chars_fpos_2)
	levenshtein_en_score = word_embedding_levenshtein(sentence_1, sentence_2, embeddings, average_distance, normalise=True)
	levenshtein_e_score = word_embedding_levenshtein(sentence_1, sentence_2, embeddings, average_distance, normalise=False)
	levenshtein_w_score = weighted_levenshtein(sentence_1, sentence_2, min_df, normalise=False)
	levenshtein_wn_score = weighted_levenshtein(sentence_1, sentence_2, min_df, normalise=True)

	jaccard_score = jaccard(sentence_1, sentence_2)
	jaccard_f_score = jaccard(filtered_sentence_1, filtered_sentence_2)

	#apted_score = apted(tree1, tree2)

	# 3. write JSON to stdout.
	tmp_results = {
		'idx': [idx1, idx2],
		'meaning_scores': {
			'cdist': float(cdist_score),
			'l2': float(l2_score),
		},
		'text_scores': {
			'levenshtein_f': levenshtein_f_score,
			'levenshtein_fn': levenshtein_fn_score,
			'levenshtein_e': levenshtein_e_score,
			'levenshtein_en': levenshtein_en_score,
			'levenshtein_w': levenshtein_w_score,
			'levenshtein_wn': levenshtein_wn_score,
			'levenshtein_fpos':levenshtein_fpos_score,
			'levenshtein': levenshtein_score,
			'levenshtein_pos':levenshtein_pos_score,
			'levenshtein_n': levenshtein_n_score,
			'jaccard': jaccard_score,
			'jaccard_f': jaccard_f_score,
			#'apted': apted_score,
		},
	}
	sys.stdout.write(json.dumps(tmp_results) + '\n')
	return tmp_results


def all_distances(sample, sentences, tree_idx, pos_decored_idx, meanings_idx, filtered_levenshtein, filtered_levenshtein_pos, w2c, remap, dump_vals, embeddings, average_distance, min_df):
	"""
		Compute all text distance and meaning distance
	"""

	# TODO: i think this condition should be removed
	if remap:
		uniq_cats = {i for p in sample for i in p}
		num_cats = len(uniq_cats)
		mapping = dict(zip(uniq_cats, random.sample(uniq_cats, num_cats)))
	# vals = []

	pool = multiprocessing.Pool(multiprocessing.cpu_count())
	single_args = (
		[sentences, tree_idx, pos_decored_idx, meanings_idx, filtered_levenshtein, filtered_levenshtein_pos, w2c, remap, dump_vals, embeddings, average_distance, min_df] + list(s)
		for s in sample
	)

	calls = pool.imap_unordered(compute_val, single_args, 200)
	for v in tqdm.tqdm(calls, total=len(sample), file=sys.stderr, leave=False):
		# vals.append(v)
		pass
	pool.close()
	pool.join()

	"""if dump_vals:
	sys.stderr.write('\ndumping scores...\n')
	with open(dump_vals, 'w') as dumpfile:
		print(vals, file=dumpfile)"""

	return


	#cdist_scores = [r['meaning_scores']['cdist'] for r in vals]
	#l2_scores = [r['meaning_scores']['l2'] for r in vals]

	#levenshtein_scores = [r['text_scores']['levenshtein'] for r in vals]
	#levenshtein_pos_scores = [r['text_scores']['levenshtein_pos'] for r in vals]
	#levenshtein_n_scores = [r['text_scores']['levenshtein_n'] for r in vals]
	#jaccard_scores = [r['text_scores']['jaccard'] for r in vals]
	#apted_scores = [r['text_scores']['apted'] for r in vals]


	#results = {}
	#for m_d, m_d_name in ((cdist_scores, 'cdist'), (l2_scores, 'l2')):
	#	for t_d, t_d_name in ((levenshtein_scores, 'levenshtein'), (levenshtein_pos_scores, 'levenshtein_pos'), (levenshtein_n_scores, 'levenshtein_n'), (jaccard_scores, 'jaccard'), (apted_scores, 'apted')):
	#		k = '%s / %s' % (m_d_name, t_d_name)
	#		v = scipy.stats.spearmanr(m_d, t_d).correlation
	#		results[k] = v

	#return results


#### Entrypoint
if __name__ == "__main__":

	p = argparse.ArgumentParser("Script to compute all pairwise distances. Outputs one JSON per pair to STDOUT, logs info to STDERR")
	#p.add_argument('--restrict_dataset_size', default=16384, type=int)
	#p.add_argument('--sample_size', default=524288, type=int)
	p.add_argument('base_file', help="dataset (definitions as TSV)")
	p.add_argument('--with_embs', required=True, type=str, help="embeddings for meaning distances")
	p.add_argument('--ft_embs', default="../cc.en.300.bin", type=str, help="fastText for Levenshtein w/ embedding weigthing")
	#p.add_argument('--annot_files', default=None, type=str)
	#p.add_argument('--baseline_support', default=30, type=int)
	p.add_argument('--average_distance', default=1.563646, type=float, help="expected distance between two embeddings for Levenshtein w/ embedding weighting")
	#p.add_argument('--output_file', default='output.json', type=str)

	args = p.parse_args()

	sys.stderr.write('\nloading models\n')

	predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
	import gensim
	embeddings = gensim.models.fasttext.load_facebook_model(args.ft_embs)
	average_distance = args.average_distance or np.array([
		np.linalg.norm(embeddings.wv[e1] - embeddings.wv[e2])
		for e1, e2 in itertools.combinations(random.sample(list(embeddings.wv.vocab), 1024), 2)
	]).mean()
	sys.stderr.write("\n%f\n" % average_distance)
	if args.with_embs:
		meanings_model = gensim.models.KeyedVectors.load_word2vec_format(args.with_embs)
		sys.stderr.write('\nreading defs\n')

		with open(args.base_file) as istr:
			import csv
			lines = list(csv.reader(istr))
			meanings, sentences = zip(*lines)
			sentences = [
				tuple([
					'%s_%i' % (w,i) if w == '<unk>' else w
					for w in s.split()
				])
				for i,s in enumerate(sentences)
			]

		meanings_idx = [meanings_model[d] for d in meanings]

		embeddings = {w:embeddings.wv[w] for s in sentences for w in s}
	#else:
	#	import torchvision
	#	torch.set_grad_enabled(False)
	#	nlp = spacy.load('en_core_web_sm')
	#	resnet = torchvision.models.resnet152(pretrained=True)
	#	resnet.eval()
	#	sys.stderr.write('\nreading captions\n')

	#	dataset = torchvision.datasets.CocoCaptions(args.base_file, args.annot_files, transform=torchvision.transforms.ToTensor())
	#	if torch.cuda.is_available():
	#		resnet = resnet.cuda()
	#		meanings_idx = [
	#			i # repeat for coindexation
	#			for img, defs in dataset
	#			for i in [resnet(img.cuda().unsqueeze(0)).view(-1).cpu().numpy()] * len(defs)
	#		]
	#		resnet = resnet.cpu()
	#	else:
	#		meanings_idx = [
	#			i # repeat for coindexation
	#			for img, defs in dataset
	#			for i in [resnet(img.unsqueeze(0)).view(-1).numpy()] * len(defs)
	#		]
	#	sentences = [
	#		tuple([
	#			str(t)
	#			for t in nlp(d)
	#		])
	#		for _, defs in dataset
	#		for d in defs
	#	]

	sys.stderr.write('\nsampling\n')

	sample = sample_pairs(sentences, sample_size=args.sample_size, restrict_dataset_size=args.restrict_dataset_size)

	sys.stderr.write('\npre-processing\n')
	forbidden_pos = {'DT', 'IN', 'CC', 'TO', '-LRB-', '-RRB-', ',', '.'}

	tree_idx = {}
	pos_decored_idx = {}
	filtered_levenshtein = {}
	filtered_levenshtein_pos = {}
	tf = {}
	N = len(sentences)
	df = collections.defaultdict(int)
	for i,s in enumerate(sentences):
		tf[i]=dict(collections.Counter(s))
		for w in set(s):
			df[w] += 1
	"""tf_idf = {
		tf[i]:{
			w: tf[i][w] * (N / df[w])
			for w in tf[i]
		}
		for i in tf
	}"""
	import string, os
	dump_tree_files = os.path.basename(args.base_file) + ".trees.tsv"
	ostr = open(dump_tree_files, "w")
	STOP_WORDS = spacy.lang.en.stop_words.STOP_WORDS | {'d', 's', "'", 're', 've', 'll', 'm'} | set(string.punctuation)
	min_df = {w: 1 - (df[w]/N) for w in df}

	def do_the_preprocess(datum):
		i, sentence = datum
		return i, predictor.predict(sentence=' '.join(sentence))

	# TODO: use enumerate
	def get_preprocess_argument():
		for i in range(len(sentences)):
			yield i, sentences[i]

	import logging
	logging.disable(sys.maxsize)
	#with multiprocessing.Pool(multiprocessing.cpu_count()) as pool: # commented out as allennlp + multiprocess seems to freeze
	for i, parsed_data in tqdm.tqdm(map(do_the_preprocess, get_preprocess_argument()), total=len(sentences), file=sys.stderr, leave=False):
			#parse = predictor.predict(sentence=' '.join(sentences[i]))
			tree_idx[i] = to_brkt(parsed_data['trees'])
			pos_decored_idx[i] = sentences[i] + tuple(parsed_data['pos_tags'])
			len_sent = len(sentences[i])
			s = tuple([sentences[i][j] for j in range(len_sent) if sentences[i][j] not in STOP_WORDS])
			p = tuple([parsed_data['pos_tags'][j] for j in range(len_sent) if sentences[i][j] not in STOP_WORDS])
			filtered_levenshtein[i] = s
			filtered_levenshtein_pos[i] = s + p

	for pair in  tqdm.tqdm(sample, file=sys.stderr, leave=False):
		print(*pair, tree_idx[pair[0]], tree_idx[pair[1]], sep="\t", file=ostr)
	"""	for i in pair:
			if not i in pos_decored_idx:
				parse = predictor.predict(sentence=' '.join(sentences[i]))
				tree_idx[i] = to_brkt(parse['trees'])
				pos_decored_idx[i] = sentences[i] + tuple(parse['pos_tags'])
				len_sent = len(sentences[i])
				s = tuple([sentences[i][j] for j in range(len_sent) if sentences[i][j] not in STOP_WORDS])
				p = tuple([parse['pos_tags'][j] for j in range(len_sent) if sentences[i][j] not in STOP_WORDS])
				filtered_levenshtein[i] = s
				filtered_levenshtein_pos[i] = s + p
		print(*pair, tree_idx[pair[0]], tree_idx[pair[1]], sep="\t", file=ostr)"""
	ostr.close()

	w2c = collections.defaultdict(itertools.count().__next__)

	sys.stderr.write('\ncomputing correlations\n')

	_ = all_distances(sample, sentences, tree_idx, pos_decored_idx, meanings_idx, filtered_levenshtein, filtered_levenshtein_pos, w2c, False, 'scores.json',  embeddings, average_distance, min_df)
	#print(json.dumps(true_score_results))


	"""pool = multiprocessing.Pool(multiprocessing.cpu_count())
	single_arg = [sentences, tree_idx, pos_decored_idx, meanings_idx, w2c, True, False]

	baseline_results = list(pool.map(correlation_fn, itertools.repeat(single_arg, args.baseline_support)))
	json_output = {'true_score_results':true_score_results, 'baseline_results':baseline_results}
	sys.stdout.write(str(json_output) + '\n')
	with open(args.output_file, "w") as ostr:
		json.dump(json_output, ostr)"""
