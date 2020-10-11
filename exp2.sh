#!/usr/bin/env bash

echo -e '\e[33m\e[1m 0. set up: activate virtual env, create required directories  \e[0m';
VENV3_ACTIVATION=.venv3/bin/activate;
source $VENV3_ACTIVATION;

EMB_W2V="data/exp2/embs/w2v/GoogleNews-vectors-negative300.bin";
EMB_FT="data/exp2/embs/ft/cc.en.300.bin";
EMB_GV6="data/exp2/embs/gv6/glove.6B.300d.txt";
EMB_GV840="data/exp2/embs/gv840/glove.840B.300d.txt";


for SCENARIO in base paraphrase; do
  for EMB_ARCH in w2v ft gv6 gv840; do
    mkdir -p data/embs_exp2/${EMB_ARCH}/${SCENARIO};
  done;
  mkdir -p data/trees_exp2/${SCENARIO} data/results_exp2/dists/${SCENARIO};
done;
mkdir -p data/results_exp2/mantels data/results_exp2/men;


echo -e '\e[33m\e[1m 1a. retrieve embeddings \e[0m';
for SCENARIO in base paraphrase; do
  echo -e "\e[33m\e[1m ${SCENARIO} \e[0m";
  echo -e '\e[33m\e[1m Word2Vec \e[0m';
  for FILE in $(find data/exp2/csv/$SCENARIO -type f); do
    OUTPUT_FILE="data/embs_exp2/w2v/$SCENARIO/$(basename ${FILE%.csv}.tsv)";
    python3 src/exp2_make_embeddings_file.py \
      --input_file $FILE \
      --output_file $OUTPUT_FILE \
      --embs_path ${EMB_W2V} \
      --is_binary;
  done;
  echo -e '\e[33m\e[1m FastText \e[0m';
  for FILE in $(find data/exp2/csv/$SCENARIO -type f); do
    OUTPUT_FILE="data/embs_exp2/ft/$SCENARIO/$(basename ${FILE%.csv}.tsv)";
    python3 src/exp2_make_embeddings_file.py \
      --input_file $FILE \
      --output_file $OUTPUT_FILE \
      --embs_path ${EMB_FT} \
      --is_fasttext;
  done;
  echo -e '\e[33m\e[1m GloVe 6B \e[0m';
  for FILE in $(find data/exp2/csv/$SCENARIO -type f); do
    OUTPUT_FILE="data/embs_exp2/gv6/$SCENARIO/$(basename ${FILE%.csv}.tsv)";
    python3 src/exp2_make_embeddings_file.py \
      --input_file $FILE \
      --output_file $OUTPUT_FILE \
      --embs_path ${EMB_GV6};
  done;
  echo -e '\e[33m\e[1m GloVe 840B \e[0m';
  for FILE in $(find data/exp2/csv/$SCENARIO -type f); do
    OUTPUT_FILE="data/embs_exp2/gv840/$SCENARIO/$(basename ${FILE%.csv}.tsv)";
    python3 src/exp2_make_embeddings_file.py \
      --input_file $FILE \
      --output_file $OUTPUT_FILE \
      --embs_path ${EMB_GV840};
  done;
done;

echo -e '\e[33m\e[1m 1c. evaluate on MEN \e[0m';
MEN_PATH="data/MEN/MEN_dataset_natural_form_full";
echo -e '\e[33m\e[1m Word2Vec \e[0m';
python3 src/exp2_test_men.py \
  --embs ${EMB_W2V} \
  --men_path ${MEN_PATH} \
  --output_file "data/results_exp2/men/w2v-results-men.txt" \
  --is_binary;
echo -e '\e[33m\e[1m FastText \e[0m';
python3 src/exp2_test_men.py \
  --embs ${EMB_FT} \
  --men_path ${MEN_PATH} \
  --output_file "data/results_exp2/men/ft-results-men.txt" \
  --is_fasttext;
echo -e '\e[33m\e[1m GloVe 6B \e[0m';
python3 src/exp2_test_men.py \
  --embs ${EMB_GV6} \
  --men_path ${MEN_PATH} \
  --output_file "data/results_exp2/men/gv6-results-men.txt";
echo -e '\e[33m\e[1m GloVe 840B \e[0m';
python3 src/exp2_test_men.py \
  --embs ${EMB_GV840} \
  --men_path ${MEN_PATH} \
  --output_file "data/results_exp2/men/gv840-results-men.txt";

echo -e '\e[33m\e[1m 2a. compute distances (meaning, Levenshtein & Jaccard) \e[0m';
## computing form distances is only required once.
EMB_ARCH='w2v';
echo -e "\e[33m\e[1m ${EMB_ARCH} \e[0m";
for EMB_FILE in $(find data/embs_exp2/${EMB_ARCH} -type f -name '*.tsv'); do
  OUTPUT_FILE="${EMB_FILE%.tsv}.json";
  echo -e "\e[33m\e[1m ${EMB_FILE} => ${OUTPUT_FILE} \e[0m";
  python3 src/shared/compute_distances.py \
    --input $EMB_FILE \
    --output ${OUTPUT_FILE};
done;

for EMB_ARCH in ft gv6 gv840; do
  echo -e "\e[33m\e[1m ${EMB_ARCH} \e[0m";
  for EMB_FILE in $(find data/embs_exp2/${EMB_ARCH} -type f -name '*.tsv'); do
    OUTPUT_FILE="${EMB_FILE%.tsv}.json";
    echo -e "\e[33m\e[1m ${EMB_FILE} => ${OUTPUT_FILE} \e[0m";
    python3 src/shared/compute_distances.py \
      --input $EMB_FILE \
      --output ${OUTPUT_FILE} \
      --meaning_only;
  done;
done;

echo -e '\e[33m\e[1m 2b. compute distances (TED) \e[0m';
for SCENARIO in base paraphrase; do
  echo -e "\e[33m\e[1m compute trees for scenario ${SCENARIO} \e[0m";
  for FILE in $(find data/exp2/csv/${SCENARIO} -type f); do
    OUTPUT_FILE="data/trees_exp2/${SCENARIO}/$(basename ${FILE%.csv}.trees.tsv)";
    echo -e "\e[33m\e[1m ${FILE} => ${OUTPUT_FILE} \e[0m";
    python3 src/exp2_compute_trees.py \
      --input_file $FILE \
      --output_file $OUTPUT_FILE;
  done;
done;
echo -e "\e[33m\e[1m compute TED \e[0m";
for FILE in $(find data/trees_exp2 -type f -name '*.trees.tsv'); do
  OUTPUT_FILE="${FILE%.tsv}.dists.tsv";
  echo -e "\e[33m\e[1m ${FILE} => ${OUTPUT_FILE} \e[0m";
  cut -f3,4 $FILE > tmp.trees;
  java -jar src/shared/apted/bin/apted.jar tmp.trees > tmp.out;
  paste $FILE tmp.out > $OUTPUT_FILE;
done;
rm tmp.trees tmp.out;

echo -e '\e[33m\e[1m 2c. patch, merge and sort everything. \e[0m';
for EMB_ARCH in w2v ft gv6 gv840; do
  python3 src/shared/merge.py \
    --sort $(find data/embs_exp2/${EMB_ARCH} -type f -name '*.json') \
    --key_meanings ${EMB_ARCH};
done;

for SCENARIO in base paraphrase; do
  for RUN in $(seq 1 5); do
    echo -e "\e[33m\e[1m merge all distances for run ${RUN}, scenario ${SCENARIO} \e[0m";
    python3 src/shared/merge.py \
      --merge $(find data/embs_exp2/ -type f -name "run-${RUN}.json" | grep $SCENARIO) \
      --merged_file tmp-merge.json;
    python3 src/shared/normalize_apted.py \
      --output_file data/results_exp2/dists/${RUN}.json \
      --tree_tsv_file data/trees_exp2/$SCENARIO/run-${RUN}.trees.dist.tsv \
      --json_file tmp-merge.json;
  done;
done;
rm tmp-merge.json;

echo -e '\e[33m\e[1m 3. compute Mantel tests \e[0m';
for SCENARIO in base paraphrase; do
  echo -e "\e[33m\e[1m ${SCENARIO} \e[0m";
  python3 src/shared/compute_mantels.py \
    --input_dir data/results_exp2/dists/${SCENARIO}/ \
    --output data/results_exp2/mantels/results-${SCENARIO}.txt;
done;

echo -e '\e[33m\e[1m 4. Perform annotation \e[0m';


deactivate;
