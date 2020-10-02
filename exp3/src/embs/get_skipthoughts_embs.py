from __future__ import print_function

import csv

import skipthoughts
import numpy as np

model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)

# /!\ that's more or less python 2 code

for i in range(1, 6):
	with open("../sentences/raw_data/run-%i.txt" % i) as istr:
		data = map(str.strip, istr)
		data = list(data)
	vectors = encoder.encode(data)
	with open("../sentences/embs/skip-thoughts/emb-%i.txt" % i, "w") as ostr:
		writer = csv.writer(ostr, delimiter="\t")
		for i, message_embedding in enumerate(np.array(vectors).tolist()):
			_ = writer.writerow([data[i], " ".join(str(x) for x in message_embedding)])
