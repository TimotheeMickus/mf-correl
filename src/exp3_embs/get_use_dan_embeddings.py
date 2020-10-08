if __name__=="__main__":

	import warnings
	warnings.simplefilter(action='ignore', category=FutureWarning)

	import os
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

	import sys
	import logging
	logging.disable(sys.maxsize)

	import argparse
	import tensorflow as tf
	import tensorflow_hub as hub
	import numpy as np
	import csv
	import pathlib

	parser = argparse.ArgumentParser("Get USE DAN embeddings for raw data")
	parser.add_argument("--input_dir", type=str, required=True, help="raw data")
	parser.add_argument("--output_dir", type=str, required=True, help="output location")
	args = parser.parse_args()

	files = pathlib.Path(args.input_dir).glob("*.txt")
	output_dir = pathlib.Path(args.output_dir)

	module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
	with tf.Session() as session:
		model = hub.load(module_url)
		init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
		session.run(init_op)
		for file_name in files:
			with open(file_name % i) as istr:
				data = map(str.strip, istr)
				data = list(data)
			message_embeddings = model(data).eval()
			output_file = output_dir / file_name.with_suffix('.emb.tsv').name
			with open(output_file, "w") as ostr:
				writer = csv.writer(ostr, delimiter="\t")
				for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
					_ = writer.writerow([data[i], " ".join(str(x) for x in message_embedding)])
