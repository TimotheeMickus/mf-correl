#!/usr/bin/env bash

echo -e '\e[41m\e[1m 0. set up: activate virtual env, create required directories  \e[0m';

VENV3_ACTIVATION=.venv3/bin/activate;

source $VENV3_ACTIVATION;

for EMB_ARCH in w2v ft gv6 gv840; do
  mkdir -p data/embs_exp2/${EMB_ARCH}/base data/embs_exp2/${EMB_ARCH}/paraphrase;
done;

mkdir -p data/results_exp2/mantels data/results_exp2/men data/results_exp2/dists;


echo -e '\e[41m\e[1m 1a. retrieve embeddings \e[0m';
EMB_PATH="data/exp2/embs/w2v/GoogleNews-vectors-negative300.bin"
echo -e '\e[41m\e[1m Word2Vec \e[0m';
for FILE in $(find data/exp2/tsv/unique-definiendum -type f); do
  OUTPUT_FILE="data/embs_exp2/${EMB_ARCH}/base/${$(basename $FILE)%.tsv}.json";
  python3 src/exp2_make_embeddings_file.py --input_file $FILE --output_file $OUTPUT_FILE --embs_path ${EMB_PATH} --is_binary;
done
for FILE in $(find data/exp2/tsv/polysemous-definiendum -type f); do
  python3 src/exp2_make_embeddings_file.py --input_file $FILE --output_file $OUTPUT_FILE --embs_path ${EMB_PATH} --is_binary;
  OUTPUT_FILE="data/embs_exp2/${EMB_ARCH}/paraphrase/${$(basename $FILE)%.tsv}.json";
done

echo -e '\e[41m\e[1m FastText \e[0m';
EMB_PATH="data/exp2/embs/ft/cc.en.300.bin"
for FILE in $(find data/exp2/tsv/unique-definiendum -type f); do
  OUTPUT_FILE="data/embs_exp2/${EMB_ARCH}/base/${$(basename $FILE)%.tsv}.json";
  python3 src/exp2_make_embeddings_file.py --input_file $FILE --output_file $OUTPUT_FILE --embs_path ${EMB_PATH} --is_fasttext;
done
for FILE in $(find data/exp2/tsv/polysemous-definiendum -type f); do
  python3 src/exp2_make_embeddings_file.py --input_file $FILE --output_file $OUTPUT_FILE --embs_path ${EMB_PATH} --is_fasttext;
  OUTPUT_FILE="data/embs_exp2/${EMB_ARCH}/paraphrase/${$(basename $FILE)%.tsv}.json";
done

echo -e '\e[41m\e[1m GloVe 6B \e[0m';
EMB_PATH="data/exp2/embs/gv6/glove.6B.300d.txt"
for FILE in $(find data/exp2/tsv/unique-definiendum -type f); do
  OUTPUT_FILE="data/embs_exp2/${EMB_ARCH}/base/${$(basename $FILE)%.tsv}.json";
  python3 src/exp2_make_embeddings_file.py --input_file $FILE --output_file $OUTPUT_FILE --emb_path $EMB_PATH;
done
for FILE in $(find data/exp2/tsv/polysemous-definiendum -type f); do
  python3 src/exp2_make_embeddings_file.py --input_file $FILE --output_file $OUTPUT_FILE --emb_path $EMB_PATH;
  OUTPUT_FILE="data/embs_exp2/${EMB_ARCH}/paraphrase/${$(basename $FILE)%.tsv}.json";
done

echo -e '\e[41m\e[1m GloVe 840B \e[0m';
EMB_PATH="data/exp2/embs/gv840/glove.840B.300d.txt"
for FILE in $(find data/exp2/tsv/unique-definiendum -type f); do
  OUTPUT_FILE="data/embs_exp2/${EMB_ARCH}/base/${$(basename $FILE)%.tsv}.json";
  python3 src/exp2_make_embeddings_file.py --input_file $FILE --output_file $OUTPUT_FILE --emb_path $EMB_PATH;
done
for FILE in $(find data/exp2/tsv/polysemous-definiendum -type f); do
  python3 src/exp2_make_embeddings_file.py --input_file $FILE --output_file $OUTPUT_FILE --emb_path $EMB_PATH;
  OUTPUT_FILE="data/embs_exp2/${EMB_ARCH}/paraphrase/${$(basename $FILE)%.tsv}.json";
done

echo -e '\e[41m\e[1m 1c. evaluate on MEN \e[0m';

echo -e '\e[41m\e[1m 2a. compute distances (meaning, Levenshtein & Jaccard) \e[0m';

echo -e '\e[41m\e[1m 2b. compute distances (TED) \e[0m';

echo -e '\e[41m\e[1m 2c. patch, merge and sort everything. \e[0m';

echo -e '\e[41m\e[1m 3. compute Mantel tests \e[0m';
