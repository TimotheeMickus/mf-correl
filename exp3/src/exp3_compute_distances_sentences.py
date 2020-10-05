# TODO: fix relative imports.
from compute_distances import *

import csv
import string
from nltk.corpus import wordnet as wn


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

def pair_to_json(pair,
	w2c = collections.defaultdict(itertools.count().__next__),
	control_synonyms=True,
	stops=spacy.lang.en.stop_words.STOP_WORDS | {'d', 's', "'", 're', 've', 'll', 'm'} | set(string.punctuation)):
	# unpack
	(i1, (s1, v1)), (i2, (s2, v2)) = pair


	# meaning scores
	meaning_scores = {"l2-USE":l2(v1, v2), "cdist-USE":cdist(v1, v2)}

	# text scores: preproc
	# convert sentences to reps
	c1, c2 = to_chr_seq(w2c, s1), to_chr_seq(w2c, s2)
	# filter stop words
	s1_f, s2_f = (w for w in s1 if w not in stops), (w for w in s2 if w not in stops)
	c1_f, c2_f = to_chr_seq(w2c, s1_f), to_chr_seq(w2c, s2_f)

	# text scores
	text_scores = {
		#"jaccard_score":jaccard(c1, c2),
		#"jaccard_f_score":jaccard(c1_f, c2_f),
		"levenshtein_score":levenshtein(c1, c2),
		"levenshtein_n_score":levenshtein_normalised(c1, c2),
		"levenshtein_f_score":levenshtein(c1_f, c2_f),
		"levenshtein_fn_score":levenshtein_normalised(c1_f, c2_f),
	}
	if control_synonyms:
		syn_s1 = list(map(default_synonym, s1))
		syn_s2 = list(map(default_synonym, s2))
		syn_c1, syn_c2 = to_chr_seq(w2c, syn_s1), to_chr_seq(w2c, syn_s2)
		syn_s1_f, syn_s2_f = (w for w in syn_s1 if w not in stops), (w for w in syn_s2 if w not in stops)
		syn_c1_f, syn_c2_f = to_chr_seq(w2c, syn_s1_f), to_chr_seq(w2c, syn_s2_f)
		text_scores.update({
			"levenshtein_syn_score":levenshtein(syn_c1, syn_c2),
			"levenshtein_syn_n_score":levenshtein_normalised(syn_c1, syn_c2),
			"levenshtein_syn_f_score":levenshtein(syn_c1_f, syn_c2_f),
			"levenshtein_syn_fn_score":levenshtein_normalised(syn_c1_f, syn_c2_f),
		})

	# return
	jdict = {
		"idx": [i1, i2],
		"sentences":[s1,s2],
		"meaning_scores":meaning_scores,
		"text_scores":text_scores,
	}
	return json.dumps(jdict)

if __name__=="__main__":
	p = argparse.ArgumentParser("""Computing distances for sentences pairs.
		Takes as input sentence embedding + tokenized sentence TSV (see embs/).
		Produces one JSON per sentence pair.""")
	p.add_argument("--input", type=str, help="input file", required=True)
	p.add_argument("--output", type=str, help="output file", default="output.json")
	args = p.parse_args()

	sample = make_pairs(read_tsv(args.input))

	with open(args.output, "w") as ostr, multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
		for jstring in tqdm.tqdm(pool.imap_unordered(pair_to_json, sample, 200), total=(4123 * 4122) // 2):
			print(jstring, file=ostr)
	pool.close()
	pool.join()
