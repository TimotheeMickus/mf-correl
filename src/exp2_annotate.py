import json
import os
import itertools
import operator
import csv
import pathlib

def read_file(filename):
	with open(filename) as istr:
		istr = map(str.strip, istr)
		istr = map(json.loads, istr)
		data = list(istr)
	return data

def get_raw_data(raw_datafile):
	# get raw data
	with open(raw_datafile) as istr:
		raw_data = csv.reader(istr)
		raw_data = list(raw_data)
	return raw_data

def get_defs(base_dir, raw_data):
	# find all files to annotate
	data = pathlib.Path(base_dir).glob("*.json")
	# open files and retrieve sequences of json dicts
	data = map(read_file, data)
	# flatten
	data = itertools.chain.from_iterable(data)
	# get sentence indices from dict
	data = map(operator.itemgetter("idx"), data)
	# flatten
	data = itertools.chain.from_iterable(data)
	# drop duplicates
	data = set(data)
	# map to base input
	data = dict(raw_data[i] for i in data)
	return data

def get_rare_sense_annots(base_dir, raw_datafile, annot_file="tmp.rare-sense"):
	#base_dir = "rnk/filtered_nonnorm"
	#raw_datafile = "dataset/datasets/tsv/unique-definiendum/dataset-rd-choice.01.tsv"
	#annot_file = "rare_sense_annots.filtered_nonnorm.tsv"

	raw_data = get_raw_data(raw_datafile)

	with open(annot_file, "w") as ostr:
		defs = get_defs(base_dir, raw_data)
		for dum, definition in defs.items() :
			msg = "Word to be defined: %s\nDefinition: %s\nIs this a rare sense? (0/1) " % (dum, definition)
			print(dum, definition, input(msg), file=ostr, sep="\t", flush=True)
	return annot_file

def get_non_systematic_definitions_annots(base_dir, raw_datafile, annot_file="tmp.non-systematic"):
	# doesn't seem to majorly impact the first three files, so i'll just drop that
	#base_dir = "rnk/filtered_nonnorm"
	#raw_datafile = "dataset/datasets/tsv/unique-definiendum/dataset-rd-choice.01.tsv"
	#annot_file = "non_systematic_definitions_annots.filtered_nonnorm.tsv"

	raw_data = get_raw_data(raw_datafile)

	data = list(pathlib.Path(base_dir).glob("*.json"))
	data = zip(data, map(read_file, data))

	with open(annot_file, "w") as ostr:
		for filename, filecontents in data:
			for j_dict in filecontents:
				sent_a, sent_b = raw_data[j_dict["idx"][0]], raw_data[j_dict["idx"][1]]
				sent_a, sent_b = "\t".join(sent_a), "\t".join(sent_b)
				msg = "\nSentence 1: %s\nSentence 2: %s\nAre these definitions non-systematic? (no: 0/yes: 1) " % (sent_a, sent_b)
				print(filename.name, sent_a, sent_b, input(msg), file=ostr, sep="\t", flush=True)
	return annot_file

def get_synonym_based_definitions_annots(base_dir, raw_datafile, annot_file="tmp.syns"):

	#base_dir = "rnk/filtered_nonnorm"
	#raw_datafile = "dataset/datasets/tsv/unique-definiendum/dataset-rd-choice.01.tsv"
	#annot_file = "synonym_based_definitions_annots.filtered_nonnorm.tsv"

	raw_data = get_raw_data(raw_datafile)

	with open(annot_file, "w") as ostr:
		defs = get_defs(base_dir, raw_data)
		for dum, definition in defs.items() :
			definientia = definition.split()
			annot = 1 if (len(definientia) == 1)\
				or (len(definientia) == 2 and definientia[0] in {"the", "a", "an"}) \
				or (definientia[:2] == ["same", "as"])\
				else 0
			print(dum, definition, annot, file=ostr, sep="\t", flush=True)
	return annot_file

def get_score_and_rank_annots(base_dir, raw_datafile, annot_file="tmp.sc-rnk"):
	#base_dir = "rnk/filtered_nonnorm"
	#raw_datafile = "dataset/datasets/tsv/unique-definiendum/dataset-rd-choice.01.tsv"
	#annot_file = "score_and_rank_annots.filtered_nonnorm.tsv"

	raw_data = get_raw_data(raw_datafile)

	data = list(pathlib.Path(base_dir).glob("*.json"))
	data = zip(data, map(read_file, data))

	with open(annot_file, "w") as ostr:
		for filename, filecontents in data:
			for j_dict in filecontents:
				ipt_a, ipt_b = raw_data[j_dict["idx"][0]], raw_data[j_dict["idx"][1]]
				sent_a, sent_b = "\t".join(ipt_a), "\t".join(ipt_b)
				m_key = next(k for k in j_dict.keys() if k.startswith("l2") or k.startswith("cdist"))
				t_key = next(k for k in j_dict.keys() if k not in {"idx", m_key})
				annot = m_key, j_dict[m_key]["ord"], j_dict[m_key]["rnk"], t_key, j_dict[t_key]["ord"], j_dict[t_key]["rnk"], j_dict["diff"]
				print(filename.name, sent_a, sent_b, *annot, file=ostr, sep="\t", flush=True)
	return annot_file



def get_similar_prefix_definitions_annots(base_dir, raw_datafile, annot_file="tmp.prfx"):
	#base_dir = "rnk/filtered_nonnorm"
	#raw_datafile = "dataset/datasets/tsv/unique-definiendum/dataset-rd-choice.01.tsv"
	#annot_file = "similar_prefix_definitions_annots.filtered_nonnorm.tsv"

	raw_data = get_raw_data(raw_datafile)

	data = list(pathlib.Path(base_dir).glob("*.json"))
	data = zip(data, map(read_file, data))

	with open(annot_file, "w") as ostr:
		for filename, filecontents in data:
			for j_dict in filecontents:
				ipt_a, ipt_b = raw_data[j_dict["idx"][0]], raw_data[j_dict["idx"][1]]
				sent_a, sent_b = "\t".join(ipt_a), "\t".join(ipt_b)
				annot = len(list(itertools.takewhile(lambda x: x[0] == x[1], zip(ipt_a[1].split(), ipt_b[1].split()))))
				print(filename.name, sent_a, sent_b, annot, file=ostr, sep="\t", flush=True)
	return annot_file


def merge_annots(annots_synonyms, annots_rare, annots_rank, annots_prefix, output_file):

	#annots_synonyms = "synonym_based_definitions_annots.filtered_nonnorm.tsv"
	with open(annots_synonyms) as istr:
		data_synonyms = list(l.strip().split("\t") for l in istr)
		data_synonyms = {tuple(l[:2]):l[-1] for l in data_synonyms}

	#annots_rare = "rare_sense_annots.filtered_nonnorm.tsv"
	with open(annots_rare) as istr:
		data_rare = list(l.strip().split("\t") for l in istr)
		data_rare = {tuple(l[:2]):l[-1] for l in data_rare}

	#annots_rank = "score_and_rank_annots.filtered_nonnorm.tsv"
	with open(annots_rank) as istr:
		data_rank = list(l.strip().split("\t") for l in istr)
		data_rank = {tuple(l[:5]):l[5:] for l in data_rank}

	#annots_prefix = "similar_prefix_definitions_annots.filtered_nonnorm.tsv"
	with open(annots_prefix) as istr:
		data_prefix = list(l.strip().split("\t") for l in istr)
		data_prefix = [[l, tuple(l[1:3]), tuple(l[3:5])] for l in data_prefix]
		data_prefix = [
			[*l, data_rare[k1], data_rare[k2], data_synonyms[k1], data_synonyms[k2], *data_rank[tuple(l[:-1])]]
			for l,k1,k2 in data_prefix
		]

	#manual_annots_file = "manual_annots.filtered_nonnorm.tsv"
	with open(output_file, "w") as ostr:
		for line in data_prefix:
			print(*line, file=ostr, sep="\t", flush=True)



if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser("Annotate definitions")
	parser.add_argument("--input_dir", type=str, required=True)
	parser.add_argument("--data_file", type=str, required=True)
	parser.add_argument("--output_file", type=str, required=True)
	args = parser.parse_args()

	annots_rare = get_rare_sense_annots(args.input_dir, args.data_file)
	#get_non_systematic_definitions_annots() # this one will not be ran due to the limited interest it brings
	annots_syns = get_synonym_based_definitions_annots(args.input_dir, args.data_file)
	annots_prefix = get_similar_prefix_definitions_annots(args.input_dir, args.data_file)
	annots_rank = get_score_and_rank_annots(args.input_dir, args.data_file)
	merge_annots(annots_syns, annots_rank, annots_prefix, args.output_file)
