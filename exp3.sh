#!/usr/bin/env bash

echo '0. set up: activate virtual env, create required directories';

VENV3_ACTIVATION=.venv/bin/activate;
VENV2_ACTIVATION=.venv2/bin/activate;

source $VENV3_ACTIVATION;

for EMB_ARCH in USE DAN infersent skipthoughts randlstm randtf; do
  mkdir -p data/exp3_${EMB_ARCH};
done;

echo '1a. compute sentence embeddings';
echo 'USE';
python3 src/exp3_embs/get_use_embeddings.py --input_dir data/exp3 --output_dir data/exp3_USE;
echo 'DAN';
python3 src/exp3_embs/get_use_dan_embeddings.py --input_dir data/exp3 --output_dir data/exp3_DAN;
echo 'infersent';
# TODO: python3 src/exp3_embs/get_infersent_embeddings.py --input_dir data/exp3 --output_dir data/exp3_infersent;
deactivate;

source $VENV2_ACTIVATION;
echo 'skipthoughts';
python2 src/exp3_embs/get_skipthoughts_embs.py --input_dir data/exp3 --output_dir data/exp3_skipthoughts;
deactivate;

source $VENV3_ACTIVATION;
echo '1b. compute random embeddings baselines';
echo 'randlstm';
python3 src/exp3_embs/get_randlstm_embs.py --input_dir  data/exp3 --output_dir data/exp3_randlstm --pickle randlstm.pkl;
echo 'randtf';
python3 src/exp3_embs/get_randtf_embs.py --input_dir  data/exp3 --output_dir data/exp3_randtf --pickle randtf.pkl;

echo '1c. evaluate on SICK';
SICK_PATH='data/SICK/SICK.txt';
for EMB_ARCH in USE DAN infersent skipthoughts; do
  echo ${EMB_ARCH};
  python3 src/exp3_test_sick.py --emb_arch ${EMB_ARCH} --sick_path ${SICK_PATH};
done;
for EMB_ARCH in randlstm randtf; do
  echo ${EMB_ARCH};
  python3 src/exp3_test_sick.py --emb_arch ${EMB_ARCH} --emb_path ${EMB_ARCH}.pkl --sick_path ${SICK_PATH};
done;

echo '2a. compute distances';
EMB_ARCH='USE';
echo ${EMB_ARCH};
for EMB_FILE in $(find data/exp3_${EMB_ARCH} -type f -name '*.emb.tsv'); do
  echo ${EMB_FILE};
  OUTPUT_FILE={EMB_FILE%.emb.tsv}.json;
  python3 src/exp3_compute_distances_sentences.py --input $EMB_FILE --output ${OUTPUT_FILE};
done

for EMB_ARCH in DAN infersent skipthoughts randlstm randtf; do
  echo ${EMB_ARCH};
  for EMB_FILE in $(find data/exp3_${EMB_ARCH} -type f -name '*.emb.tsv'); do
    echo ${EMB_FILE};
    OUTPUT_FILE={EMB_FILE%.emb.tsv}.json;
    python3 src/exp3_compute_distances_sentences.py --input $EMB_FILE --output ${OUTPUT_FILE} --meaning_only;
  done
done;

echo '2b. patch, merge and sort everything.';
# TODO

echo '3. compute Mantel exp3_test_sick';
# TODO

deactivate;