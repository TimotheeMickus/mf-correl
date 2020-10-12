#!/usr/bin/env bash

echo -e '\e[33m\e[1m 0. set up: activate virtual env, create required directories  \e[0m';

VENV3_ACTIVATION=.venv3/bin/activate;

source $VENV3_ACTIVATION;
mkdir -p data/exp1/ data/results_exp1/

echo -e '\e[33m\e[1m  1. generate artifical languages. \e[0m';
python3 src/exp1_generate_artificial_languages.py;

echo -e '\e[33m\e[1m  2. compute distances and run mantel tests \e[0m';
python3 src/exp1_compute_distances_and_mantels.py \
  --input_dir data/exp1 \
  --output_file data/results_exp1/mantels.txt;

deactivate;
