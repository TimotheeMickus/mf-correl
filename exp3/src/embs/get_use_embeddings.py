import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv


module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
with tf.Session() as session:
	model = hub.load(module_url)
	init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
	session.run(init_op)
	for i in range(1, 6):
		with open("sentences/raw_data/run-%i.txt" % i) as istr:
			data = map(str.strip, istr)
			data = list(data)
		message_embeddings = model(data).eval()
		with open("sentences/embs/USE-transformer/emb-%i.txt" % i, "w") as ostr:
			writer = csv.writer(ostr, delimiter="\t")
			for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
				_ = writer.writerow([data[i], " ".join(str(x) for x in message_embedding)])
