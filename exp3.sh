#!/usr/bin/env bash

echo '0. set up: activate virtual env, create required directories';

VENV3_ACTIVATION=.venv3/bin/activate;
VENV2_ACTIVATION=.venv2/bin/activate;

source $VENV3_ACTIVATION;

for EMB_ARCH in USE DAN infersent skipthoughts randlstm randtf; do
  mkdir -p data/embs_exp3/${EMB_ARCH};
done;

echo '1a. compute sentence embeddings';
echo 'USE';
python3 src/exp3_embs/get_use_embeddings.py --input_dir data/exp3 --output_dir data/embs_exp3/USE;
echo 'DAN';
python3 src/exp3_embs/get_use_dan_embeddings.py --input_dir data/exp3 --output_dir data/embs_exp3/DAN;
echo 'infersent';
python3 src/exp3_embs/get_infersent_embeddings.py --input_dir data/exp3 --output_dir data/embs_exp3/infersent;
deactivate;

source $VENV2_ACTIVATION;
echo 'skipthoughts';
python2 src/exp3_embs/get_skipthoughts_embeddings.py --input_dir data/exp3 --output_dir data/embs_exp3/skipthoughts;
deactivate;

source $VENV3_ACTIVATION;
echo '1b. compute random embeddings baselines';
echo 'randlstm';
python3 src/exp3_embs/get_randlstm_embeddings.py --input_dir  data/exp3 --output_dir data/embs_exp3/randlstm --pickle randlstm.pkl;
echo 'randtf';
python3 src/exp3_embs/get_randtf_embeddings.py --input_dir  data/exp3 --output_dir data/embs_exp3/randtf --pickle randtf.pkl;

echo '1c. evaluate on SICK';
SICK_PATH='data/SICK/SICK.txt';
for EMB_ARCH in USE DAN infersent; do
  echo ${EMB_ARCH};
  python3 src/exp3_test_sick.py --emb_arch ${EMB_ARCH} --sick_path ${SICK_PATH};
done;
deactivate;

source $VENV2_ACTIVATION;
EMB_ARCH=skipthoughts;
echo ${EMB_ARCH};
python2 src/exp3_test_sick.py --emb_arch ${EMB_ARCH} --sick_path ${SICK_PATH};
deactivate;

source $VENV3_ACTIVATION;
for EMB_ARCH in randlstm randtf; do
  echo ${EMB_ARCH};
  python3 src/exp3_test_sick.py --emb_arch ${EMB_ARCH} --emb_path ${EMB_ARCH}.pkl --sick_path ${SICK_PATH};
done;

echo '2a. compute distances';
# computing form distances is only required once.
EMB_ARCH='USE';
echo ${EMB_ARCH};
for EMB_FILE in $(find data/embs_exp3/${EMB_ARCH} -type f -name '*.emb.tsv'); do
  echo ${EMB_FILE};
  OUTPUT_FILE={EMB_FILE%.emb.tsv}.json;
  python3 src/exp3_compute_distances_sentences.py --input $EMB_FILE --output ${OUTPUT_FILE};
done

for EMB_ARCH in DAN infersent skipthoughts randlstm randtf; do
  echo ${EMB_ARCH};
  for EMB_FILE in $(find data/embs_exp3/${EMB_ARCH} -type f -name '*.emb.tsv'); do
    echo ${EMB_FILE};
    OUTPUT_FILE={EMB_FILE%.emb.tsv}.json;
    python3 src/exp3_compute_distances_sentences.py --input $EMB_FILE --output ${OUTPUT_FILE} --meaning_only;
  done
done;

echo '2b. patch, merge and sort everything.';
# everything must be sorted
for EMB_ARCH in USE DAN infersent skipthoughts randlstm randtf; do
  echo "sort ${EMB_ARCH} output JSONs";
  python3 src/shared/merge.py --sort $(find data/embs_exp3/${EMB_ARCH} -type f -name '*.json') --key_meanings ${EMB_ARCH};
done;
# merge results per run
mkdir -p data/results_exp3;
for RUN in $(seq 1 5); do
  echo "merge all results for run ${RUN}";
  python3 src/shared/merge.py --merge $(find data/embs_exp3/ -type f -name "run-${RUN}.json" ) --merged_file data/results_exp3/${RUN}.json;
done;

echo '3. compute Mantel tests';
python3 src/shared/compute_mantels_per_dir.py --input_dir data/results_exp3 --output exp3-results-mantels.txt;

deactivate;
