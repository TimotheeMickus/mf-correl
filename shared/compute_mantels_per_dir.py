import json
import os
import numpy as np
import MantelTest.Mantel as mantel
import datetime
import itertools
from multiprocessing import Pool, cpu_count

def get_raw_data(filename):
	with open(filename) as istr:
		for j in map(json.loads, istr):
			yield j

def extract(filename, type_, field):
	for json_item in get_raw_data(filename):
		yield json_item[type_][field]

def mantel_from_config(arg):
	meaning_distance, form_distance, raw_datafile, method = arg
	print(datetime.datetime.now(), "extraction start for thread:", *arg)
	m_dist = np.array(list(extract(raw_datafile, 'meaning_scores', meaning_distance)))
	f_dist = np.array(list(extract(raw_datafile, 'text_scores', form_distance)))
	print(datetime.datetime.now(), "extraction complete for thread:", *arg)
	return mantel.test(m_dist, f_dist, perms=1000, method=method), arg


if __name__=="__main__":
	import argparse
	p = argparse.ArgumentParser("Compute all mantels given one directory")
	p.add_argument("--input_dir", type=str, help="input directory", required=True)
	p.add_argument("--output", type=str, help="output file", default="mantels.tsv")
	p.add_argument("--mdists", type=str, nargs='+', help="restrict to these meaning distances", default=[])
	p.add_argument("--tdists", type=str, nargs='+', help="restrict to these text distances", default=[])
	p.add_argument("--corr", type=str, choices=['spearman', 'pearson'], help="correlation computed", default="spearman")
	args = p.parse_args()
	print(datetime.datetime.now(), "process start")
	input_files = (
		os.path.join(root, filename)
		for root, _, filenames in os.walk(args.input_dir) 
		for filename in filenames 
		if filename.endswith(".json")
	)

	#define config iterator
	def get_config(mscores=[], tscores=[], method="spearman"):
		for input_file in sorted(input_files):
			ex = next(get_raw_data(input_file))
			#print(datetime.datetime.now(), "handling file %s" % input_file) 
			for meaning_distance in mscores or ex['meaning_scores']:
				for form_distance in tscores or ex['text_scores']:
					yield meaning_distance, form_distance, input_file, method

	with open(args.output, "w") as ostr, Pool(processes=cpu_count()) as pl:
		configs = get_config(mscores=args.mdists, tscores=args.tdists, method=args.corr)
		for mantel_result, config in pl.imap_unordered(mantel_from_config, configs):
			print(datetime.datetime.now(), *config, *mantel_result)
			print(*config, *mantel_result, sep="\t", file=ostr)
	
	
	pl.close()
	pl.join()

