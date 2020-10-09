import json
import operator

def get_raw_data(filename, drop={}, key_meanings=None):
	with open(filename) as istr:
		for j in map(json.loads, istr):
			#j['idx'] = list(map(int, j['idx']))
			for key in drop:
				del j[key]
			if key_meanings and "meaning_scores" in j:
				keys = list(j["meaning_scores"].keys())
				for ms in keys:
					j["meaning_scores"]["%s-%s" % (ms, key_meanings)] = j["meaning_scores"][ms]
				for ms in keys:
					del j["meaning_scores"][ms]
			yield j

def sort_file(filename, drop={}, key_meanings=None):
	data = sorted(get_raw_data(filename, drop=drop, key_meanings=key_meanings), key=lambda j:j["idx"])
	with open(filename, "w") as ostr:
		for j in map(json.dumps, data):
			print(j, file=ostr)

def _errmsg(jdicts):
	get_idx = operator.itemgetter("idx")
	return "files were not sorted.\nIndices: %s\nRaw data: %s\n" % (str(set(map(tuple, map(get_idx, jdicts)))), str(jdicts))

def merge(output_file, *files):
	get_idx = operator.itemgetter("idx")
	with open(output_file, "w") as ostr:
		for jdicts in zip(*map(get_raw_data, files)):
			assert len(set(map(tuple, map(get_idx, jdicts)))) == 1, _errmsg(jdicts) #make sure all files are sorted

			meaning_scores = {}
			for j_dict in jdicts:
				meaning_scores.update(j_dict.get("meaning_scores", {}))

			text_scores = {}
			for j_dict in jdicts:
				text_scores.update(j_dict.get("text_scores", {}))

			merged_j = {
				"idx":jdicts[0]["idx"],
				"meaning_scores":meaning_scores,
				"text_scores":text_scores,
			}
			print(json.dumps(merged_j), file=ostr)

def merge2(output_file, *files):
	get_idx = operator.itemgetter("idx")
	with open(output_file, "w") as ostr:
		for jdicts in zip(*map(get_raw_data, files)):
			assert len(set(map(tuple, map(get_idx, jdicts)))) == 1, _errmsg(jdicts) #make sure all files are sorted
			jdict_base, j_dict_supm = jdicts

			jdict_base["meaning_scores"].update(j_dict_supm.get("meaning_scores", {}))

			print(json.dumps(jdict_base), file=ostr)

def merge_meanings_and_texts(output_file, meaning_file, text_file):

	with open(output_file, "w") as ostr:
		for jdict_m, jdict_t in zip(get_raw_data(meaning_file), get_raw_data(text_file)):
			assert jdict_m["idx"] == jdict_t["idx"], _errmsg([jdict_m, jdict_t]) #make sure all files are sorted

			meaning_scores = jdict_m["meaning_scores"]

			text_scores = jdict_t["text_scores"]

			merged_j = {
				"idx":jdict_m["idx"],
				"meaning_scores":meaning_scores,
				"text_scores":text_scores,
			}
			print(json.dumps(merged_j), file=ostr)


if __name__=="__main__":
	import argparse
	import datetime
	parser = argparse.ArgumentParser("Utility script for merging different JSON distance files")
	parser.add_argument("--sort", nargs="+", type=str, default=[])
	parser.add_argument("--drop", nargs="+", type=str, default=[])
	parser.add_argument("--merge", nargs="+", type=str, default=[])
	parser.add_argument("--merge2", nargs="+", type=str, default=[])
	parser.add_argument("--key_meanings", type=str, default=None)
	parser.add_argument("--meanings_from", type=str, default=None)
	parser.add_argument("--texts_from", type=str, default=None)
	parser.add_argument("--merged_file", type=str, default="merged.json")
	args = parser.parse_args()
	for filename in args.sort:
		print(datetime.datetime.now(), "handling %s" % filename)
		sort_file(filename, drop=args.drop, key_meanings=args.key_meanings)
	if args.meanings_from and args.texts_from:
		merge_meanings_and_texts(args.merged_file, args.meanings_from, args.texts_from)
	elif args.merge:
		merge(args.merged_file, *args.merge)
	elif args.merge2:
		merge2(args.merged_file, *args.merge2)
