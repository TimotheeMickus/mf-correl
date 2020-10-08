if __name__ == "__main__":
    import argparse
    import pathlib
    import os
	parser = argparse.ArgumentParser("Get USE embeddings for raw data")
	parser.add_argument("--input_dir", type=str, required=True, help="raw data")
	parser.add_argument("--output_dir", type=str, required=True, help="output location")
	args = parser.parse_args()

    files = pathlib.Path(args.input_dir).glob("*.txt")
    output_dir = pathlib.Path(args.output_dir)

    from InferSent.models import InferSent
	model_version = 1
    dirname =  os.path.dirname(__file__)
	MODEL_PATH = os.path.join(dirname, "InferSent/encoder/infersent%s.pkl" % model_version)
	params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
		        'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
	model = InferSent(params_model)
	model.load_state_dict(torch.load(MODEL_PATH))
	W2V_PATH = os.path.join(dirname, 'InferSent/embs/glove.840B.300d.txt')
	model.set_w2v_path(W2V_PATH)
	model.build_vocab_k_words(K=100000)

    for file_name in files:
        with open(file_name) as istr:
            data = map(str.strip, istr)
            data = list(data)
        output_file = output_dir / file_name.with_suffix('.emb.tsv').name
        embeddings = infersent.encode(data, tokenize=True)
        with open(output_file, "w") as ostr:
            writer = csv.writer(ostr, delimiter="\t")
            for sent, emb in zip(data, embeddings):
                _ = writer.writerow([sent, emb.tolist()])
