#!/usr/bin/env bash

echo '0. set up: activate virtual env, create required directories';

VENV3_ACTIVATION=.venv3/bin/activate;

source $VENV3_ACTIVATION;
mkdir -p data/exp1/ data/results_exp1/

echo '1. generate artifical languages.';
python3 src/exp1_generate_artificial_languages.py;

echo '2. compute distances and run mantel tests';
python3 src/exp1_compute_distances_and_mantels.py \
  --input_dir data/exp1 \
  --output_file data/results_exp1/mantels.txt;

deactivate;
