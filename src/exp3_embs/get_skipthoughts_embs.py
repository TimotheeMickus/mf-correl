from __future__ import print_function
# /!\ that's  python 2 code
if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser("retrieve skipthoughts embs")
	parser.add_argument("--input_dir", required=True, type=str)
	parser.add_argument("--output_dir", required=True, type=str)
	args = parser.parse_args()

	import csv
	import os

	# YOU MIGHT NEED TO MODIFY THIS TO FIT YOUR INSTALL.
	from skip_thoughts.skipthoughts import load_model, Encoder

	import numpy as np

	model = load_model()
	encoder = Encoder(model)



	for i in range(1, 6):
		file_name = os.path.join(args.input_dir, "run-%i.txt" % i)
		with open(file_name) as istr:
			data = map(str.strip, istr)
			data = list(data)
		vectors = encoder.encode(data)
		output_file = os.path.join(args.output_dir, "run-%i.emb.tsv" % i)
		with open(output_file, "w") as ostr:
			writer = csv.writer(ostr, delimiter="\t")
			for i, message_embedding in enumerate(np.array(vectors).tolist()):
				_ = writer.writerow([data[i], " ".join(str(x) for x in message_embedding)])
