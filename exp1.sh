#!/usr/bin/env bash

echo '0. set up: activate virtual env, create required directories';

VENV3_ACTIVATION=.venv3/bin/activate;

source $VENV3_ACTIVATION;

echo '1. generate artifical languages.';
python3 src/exp1_generate_artifial_languages.py;

echo '2. compute distances and run mantel tests';
# TODO

deactivate;
