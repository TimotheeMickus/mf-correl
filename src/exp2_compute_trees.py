import collections

from allennlp.predictors.predictor import Predictor

import argparse
import csv
import tqdm


def to_brkt(tree):
	"""
		Convert AllenNLP square bracket tree format to apted curly bracket tree format
	"""
	prep = tree.replace('(', '{').replace(')', '}')
	prep = re.sub(r" ([^{} ]+)}",r" {\1}}", prep)
	return prep.replace(' ', '')


if __name__ == "__main__":

	p = argparse.ArgumentParser("Precompute trees for APTED.")
	p.add_argument('--input_file', type=str, required=True, help="dataset (definitions as TSV)")
	p.add_argument('--output_file', type=str, required=True, help="path to output trees")
	args = p.parse_args()

	predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
	with open(args.input_file) as istr:
			import csv
			lines = list(csv.reader(istr, delimiter="\t"))
			_, sentences = zip(*lines)

	sample = itertools.combinations(range(len(sentences)), 2)
	trees = collections.defaultdict(lambda i: to_brkt(predictor.predict(sentence=' '.join(sentences[i]))['trees']))
	with open(args.output_file, "w") as ostr:
		for a, b in tqdm.tqdm(sample, total=(len(sentences) * (len(sentences) - 1)) / 2):
			print(a, b, trees[a], trees[b], sep="\t", file=ostr)
