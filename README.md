# What Meaning-Form Correlation Has to Compose With
This is the repository for the [COLING 2020](https://coling2020.org/) paper ["What Meaning-Form Correlation Has to Compose With"](nowhere.org/place-holder).


## Installing

An installation script is available to help you set up all external dependencies as required by the code: [`install.sh`](install.sh)
The process assumes a UNIX environment and access to functional installs of python 3.7 and python 2 as well as the `virtualenv` tool. Running the code also requires a working Java environment (tested with OpenJDK 11.0.6 2020-01-14).

**NB: The installation script will download all dependencies, which might require significant space.**

## Structure

Code is stored in `src/`. The `src/shared/` directory contains pieces of code shared by some or all experiments. Script starting with `src/exp1_`, `src/exp2_` and `src/exp3_` correspond to code for artificial language experiments, definition experiments and sentence experiments respectively. Subdirectory `src/exp3_embs/` contains specifically scripts to compute or retrieve sentence embeddings.

Data is available under `data/`, subdirectories correspond to different experiments.

"Push-button" scripts are available to reproduce experiments: [`exp1.sh`](exp1.sh), [`exp2.sh`](exp2.sh), [`exp3.sh`](exp3.sh).

**NB: Temp files produced for experiments 2 & 3 are very large (> 150Gb). Consider running part of the experiments or make sure you have dedicated free space.**

## Acknowledgments

We used Mantel tests from J. W. Carr's github ([see here](https://github.com/jwcarr/MantelTest)).

The implementation of APTED by Pawlik & Augsten is from their original [repository](https://github.com/DatabaseGroup/apted) (written in Java).
The JAR we provide has been hacked to accept a file of pairs of trees at once, instead of a single pair.
The single edited Java file is available for reference in the directory `shared/apted/`.

Pre-trained word embeddings are available from their original repositories, or at these links: [Word2Vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing), [GloVe 6B](http://nlp.stanford.edu/data/glove.6B.zip) and [840B](http://nlp.stanford.edu/data/glove.840B.300d.zip), [FastText](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz)

Sentence encoders are from the original repositories: [SkipThoughts](https://github.com/ryankiros/skip-thoughts) (written in python 2), [InferSent](https://github.com/facebookresearch/InferSent).
See also the original Google Hub for [USE DAN](https://tfhub.dev/google/universal-sentence-encoder/4) and [USE Transformer](https://tfhub.dev/google/universal-sentence-encoder-large/5).


Lastly, the two datasets used to evaluate embeddings, the [MEN](https://staff.fnwi.uva.nl/e.bruni/MEN) by Bruni et al. and [SICK](http://marcobaroni.org/composes/sick.html) by Baroni et al. can be retrieved from their respective homepages.
