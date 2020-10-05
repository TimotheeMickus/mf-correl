from datetime import datetime
import json
import itertools

filename = "/home/tmickus/mf-correl/runs/unique/run-01/01.patched.json.ranked"

with open(filename) as istr:
	data = list(map(json.loads, istr))
texts = list(data[0]["text_scores"].keys())
meanings = list(data[0]["meaning_scores"].keys())

def get_rank_diff(data, i, t, m):
	return abs(data[i]["text_scores"][t] - data[i]["meaning_scores"][m])

for text, meaning in itertools.product(texts, meanings):
	print(datetime.now(), text, meaning)
	data.sort(key=lambda j: abs(j["text_scores"][text] - j["meaning_scores"][meaning]), reverse=True)
	collected_cases = []
	i = 0
	while len(collected_cases) < 100:
		current_value = get_rank_diff(data, i, text, meaning)
		while get_rank_diff(data, i, text, meaning) == current_value:
			collected_cases.append(data[i])
			i += 1
	with open("at-least-1C.%s-%s.uniq.run-01.json" % (text, meaning), "w") as ostr:
		for line in map(json.dumps, collected_cases):
			print(line, file=ostr)
