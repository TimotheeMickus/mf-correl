import json, itertools

def c2i(c):
	if c == '{': return 1
	if c == '}': return -1
	return 0

def height(tree_str):
	return max(itertools.accumulate(map(c2i, tree_str)))

def nnodes(tree_str):
	return sum(c == "{" for c in tree_str)

def normalisation(tree1, tree2):
	return nnodes(tree1) + nnodes(tree2) - min(height(tree1), height(tree2))


def merge_using_apted(tree_tsv_filename, json_filename, output_filename):
	with open(tree_tsv_filename) as tree_istr, open(json_filename) as json_istr, open(output_filename, "w") as ostr:
		tree_istr = (l.strip().split("\t") for l in tree_istr)
		json_istr = map(json.loads, json_istr)
		for tree_record, j_dict in zip(tree_istr, json_istr):
			assert list(map(int, tree_record[:2])) == j_dict["idx"], "mismatched files"
			n = normalisation(*tree_record[2:-1])
			j_dict["text_scores"]["apted"] = float(tree_record[-1])
			j_dict["text_scores"]["apted_n"] = j_dict["text_scores"]["apted"] / n
			print(json.dumps(j_dict), file=ostr)
			

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser("utility script to normalize apted scores.")
	parser.add_argument("--json_file", type=str, required=True)
	parser.add_argument("--tree_tsv_file", type=str, required=True)
	parser.add_argument("--output_file", type=str, required=True)
	args = parser.parse_args()
	merge_using_apted(args.tree_tsv_file, args.json_file, args.output_file)
