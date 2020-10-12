import json
import numpy as np
from scipy.stats import rankdata
import datetime
from multiprocessing import Pool, cpu_count
import tqdm
import collections
import sys, itertools, os, shutil, functools
import math, pathlib, pickle


def get_raw_data(filename):
	# read jsons
	with open(filename) as istr:
		for j in map(json.loads, istr):
			yield j


def read(filename, score_type, dist):
	# read interesting field
	for json_item in get_raw_data(filename):
		yield json_item[score_type][dist]


def do_ranking(filename, score_type, dist, prec = 10**6):
	# convert to ranks
	idx = (tuple(j["idx"]) for j in get_raw_data(filename))
	vals = read(filename, score_type, dist)
	vals = (math.ceil(v * prec) for v in vals) # round to some precision and cast to int
	idx, vals = zip(*sorted(zip(idx, vals), key=lambda p:p[-1]))

	fst, snd = itertools.tee(vals)
	_ = next(snd) # offset by one
	new_rank = itertools.chain([True], (f != s for f,s in zip(fst, snd)))
	rank = []
	for i, new_rnk in enumerate(new_rank):
		if new_rnk:
			current_value = i + 1
		rank.append(current_value)

	vals = sorted(zip(idx, rank), key=lambda p:p[0])
	tmp_dir = pathlib.Path(".tmp/%s/%s" % (score_type, dist))
	tmp_dir.mkdir(parents=True, exist_ok=True)
	with open(tmp_dir / "vals.pkl", "wb") as ostr:
		pickle.dump(vals, ostr)

def do_ranking_single_arg(single_arg):
	return do_ranking(*single_arg)


def _cpt_rnk_dif(p):
	# util for computing rank diff
	(ord_m, (idx_m, rnk_m), ord_t, (idx_t, rnk_t)) = p
	return abs(rnk_m - rnk_t)


def extract_worst(filename, meaning, text, output_prefix, num=100):
	# getting worst rank diff data
	ordinals_m = read(filename, "meaning_scores", meaning)
	ordinals_t = read(filename, "text_scores", text)

	with open(".tmp/%s/%s/vals.pkl" % ("meaning_scores", meaning), "rb") as istr:
		meanings_vals = pickle.load(istr)

	with open(".tmp/%s/%s/vals.pkl" % ("text_scores", text), "rb") as istr:
		texts_vals = pickle.load(istr)

	vals = zip(ordinals_m, meanings_vals, ordinals_t, texts_vals)
	vals = sorted(vals, key=_cpt_rnk_dif, reverse=True)
	worst = []
	for p in vals:
		(ord_m, (idx_m, rnk_m), ord_t, (idx_t, rnk_t)) = p
		rnk_diff = _cpt_rnk_dif(p)
		assert idx_m == idx_t, "sorting problem: " + str(idx_t) + " " + str(idx_m) + " " + text + " " + meaning
		j_dict = {
			"idx":idx_m,
			meaning:{
				"ord":ord_m,
				"rnk":rnk_m,
			},
			text:{
				"ord":ord_t,
				"rnk":rnk_t,
			},
			"diff":rnk_diff,
		}
		if len(worst) < num:
			worst.append(j_dict)
		elif worst[-1]["diff"] == rnk_diff:
			worst.append(j_dict)
		else:
			break

	with open("%s_%s_%s_%s.json" % (output_prefix, os.path.basename(filename), meaning, text), "w") as ostr:
		for j_dict in worst:
			print(json.dumps(j_dict), file=ostr)
	return worst


def extract_single_arg(single_arg):
	# wrap call in single arg
	return extract_worst(*single_arg)


def get_configs_for_ranking(files, mdist=[], tdist=[]):
	# loop 1 for precomputing rankings
	for filename in files:
		ex = next(get_raw_data(filename))
		for meaning in mdist or ex["meaning_scores"]:
			yield filename, "meaning_scores", meaning
		for text in tdist or ex["text_scores"]:
			yield filename, "text_scores", text


def _cnt_cfgs_rnk(files, mdist=[], tdist=[]):
	# number of loop 1 iters
	total = 0
	for filename in files:
		ex = next(get_raw_data(filename))
		for meaning in mdist or ex["meaning_scores"]:
			total += 1
		for text in tdist or ex["text_scores"]:
			total += 1
	return total


def get_configs_for_extract(files, output_prefix, mdist=[], tdist=[]):
	# loop 2 for extracting 100 worst
	for filename in files:
		ex = next(get_raw_data(filename))
		for meaning in mdist or ex["meaning_scores"]:
			for text in tdist or ex["text_scores"]:
				yield filename, meaning, text, output_prefix


def _cnt_cfgs_extr(files, mdist=[], tdist=[]):
	# number of loop 2 iters
	total = 0
	for filename in files:
		ex = next(get_raw_data(filename))
		for meaning in mdist or ex["meaning_scores"]:
			for text in tdist or ex["text_scores"]:
				total += 1
	return total


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser("producing ranks")
	parser.add_argument("--files", nargs='+', type=str, default=[])
	parser.add_argument("--tdists", nargs='+', type=str, default=[])
	parser.add_argument("--mdists", nargs='+', type=str, default=[])
	parser.add_argument("--output_prefix", type=str, default="rankdiff")
	args = parser.parse_args()


	with Pool(cpu_count()) as pool:
		# loop 1
		print(datetime.datetime.now(), "pre-ranking")

		configs = list(get_configs_for_ranking(args.files, mdist=args.mdists, tdist=args.tdists))
		n_configs = _cnt_cfgs_rnk(args.files, mdist=args.mdists, tdist=args.tdists)

		for _ in tqdm.tqdm(pool.imap_unordered(do_ranking_single_arg, configs), total=n_configs):
			pass

	with Pool(cpu_count()) as pool:
		# loop 2
		print(datetime.datetime.now(), "extracting worst items")

		configs = get_configs_for_extract(args.files, args.output_prefix, mdist=args.mdists, tdist=args.tdists)
		n_configs = _cnt_cfgs_extr(args.files, mdist=args.mdists, tdist=args.tdists)

		for _ in tqdm.tqdm(pool.imap_unordered(extract_single_arg, configs), total=n_configs):
			pass
	print(datetime.datetime.now(), "all done.")
