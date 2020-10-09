#!/usr/bin/env bash

echo -e '\e[41m\e[1m 0. set up: activate virtual env, create required directories  \e[0m';

VENV3_ACTIVATION=.venv3/bin/activate;
VENV2_ACTIVATION=.venv2/bin/activate;

source $VENV3_ACTIVATION;

for EMB_ARCH in USE DAN infersent skipthoughts randlstm randtf; do
  mkdir -p data/embs_exp3/${EMB_ARCH};
done;
mkdir -p data/results_exp3/mantels data/results_exp3/sick data/results_exp3/dist;


echo -e '\e[41m\e[1m 1a. compute sentence embeddings \e[0m';
#
echo -e '\e[41m\e[1m USE \e[0m';
python3 src/exp3_embs/get_use_embeddings.py --input_dir data/exp3 --output_dir data/embs_exp3/USE;
echo -e '\e[41m\e[1m DAN \e[0m';
python3 src/exp3_embs/get_use_dan_embeddings.py --input_dir data/exp3 --output_dir data/embs_exp3/DAN;
echo -e '\e[41m\e[1m infersent \e[0m';
python3 src/exp3_embs/get_infersent_embeddings.py --input_dir data/exp3 --output_dir data/embs_exp3/infersent;
deactivate;

source $VENV2_ACTIVATION;
echo -e '\e[41m\e[1m skipthoughts \e[0m';
python2 src/exp3_embs/get_skipthoughts_embeddings.py --input_dir data/exp3 --output_dir data/embs_exp3/skipthoughts;
deactivate;

source $VENV3_ACTIVATION;
echo -e '\e[41m\e[1m 1b. compute random embeddings baselines \e[0m';
echo -e '\e[41m\e[1m randlstm \e[0m';
python3 src/exp3_embs/get_randlstm_embeddings.py --input_dir  data/exp3 --output_dir data/embs_exp3/randlstm --pickle data/embs_exp3/randlstm.pkl;
echo -e '\e[41m\e[1m randtf \e[0m';
python3 src/exp3_embs/get_randtf_embeddings.py --input_dir  data/exp3 --output_dir data/embs_exp3/randtf --pickle data/embs_exp3/randtf.pkl;

echo -e '\e[41m\e[1m 1c. evaluate on SICK \e[0m';
SICK_PATH='data/SICK/SICK.txt';
for EMB_ARCH in USE DAN infersent; do
  echo -e "\e[41m\e[1m ${EMB_ARCH} \e[0m";
  python3 src/exp3_test_sick.py --emb_arch ${EMB_ARCH} --sick_path ${SICK_PATH} --output_file "data/results_exp3/sick/${EMB_ARCH}-results-sick.txt";
done;
deactivate;

source $VENV2_ACTIVATION;
EMB_ARCH=skipthoughts;
echo -e "\e[41m\e[1m ${EMB_ARCH} \e[0m";
python2 src/exp3_test_sick.py --emb_arch ${EMB_ARCH} --sick_path ${SICK_PATH} --output_file "data/results_exp3/sick/${EMB_ARCH}-results-sick.txt";
deactivate;

source $VENV3_ACTIVATION;
for EMB_ARCH in randlstm randtf; do
  echo -e "\e[41m\e[1m ${EMB_ARCH} \e[0m";
  python3 src/exp3_test_sick.py --emb_arch ${EMB_ARCH} --emb_path data/embs_exp3/${EMB_ARCH}.pkl --sick_path ${SICK_PATH} --output_file "data/results_exp3/sick/${EMB_ARCH}-results-sick.txt";
done;

echo -e '\e[41m\e[1m 2a. compute distances \e[0m';
## computing form distances is only required once.
EMB_ARCH='USE';
echo -e "\e[41m\e[1m ${EMB_ARCH} \e[0m";
for EMB_FILE in $(find data/embs_exp3/${EMB_ARCH} -type f -name '*.emb.tsv'); do
  OUTPUT_FILE="${EMB_FILE%.emb.tsv}.json";
  echo -e "\e[41m\e[1m ${EMB_FILE} to ${OUTPUT_FILE} \e[0m";
  python3 src/exp3_compute_distances_sentences.py --input $EMB_FILE --output ${OUTPUT_FILE};
done

for EMB_ARCH in DAN infersent skipthoughts randlstm randtf; do
  echo -e "\e[41m\e[1m ${EMB_ARCH} \e[0m";
  for EMB_FILE in $(find data/embs_exp3/${EMB_ARCH} -type f -name '*.emb.tsv'); do
    OUTPUT_FILE="${EMB_FILE%.emb.tsv}.json";
    echo -e "\e[41m\e[1m ${EMB_FILE} to ${OUTPUT_FILE} \e[0m";
    python3 src/exp3_compute_distances_sentences.py --input $EMB_FILE --output ${OUTPUT_FILE} --meaning_only;
  done
done;

echo -e '\e[41m\e[1m 2b. patch, merge and sort everything. \e[0m';
## everything must be sorted
for EMB_ARCH in USE DAN infersent skipthoughts randlstm randtf; do
  echo -e "\e[41m\e[1m sort ${EMB_ARCH} output JSONs \e[0m";
  python3 src/shared/merge.py --sort $(find data/embs_exp3/${EMB_ARCH} -type f -name '*.json') --key_meanings ${EMB_ARCH};
done;
## merge results per run
for RUN in $(seq 1 5); do
  echo -e "\e[41m\e[1m merge all results for run ${RUN} \e[0m";
  python3 src/shared/merge.py --merge $(find data/embs_exp3/ -type f -name "run-${RUN}.json" ) --merged_file data/results_exp3/dists/${RUN}.json;
done;

echo -e '\e[41m\e[1m 3. compute Mantel tests \e[0m';
python3 src/shared/compute_mantels_per_dir.py --input_dir data/results_exp3/dists/ --output data/results_exp3/mantels/results.txt;

deactivate;
