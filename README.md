# What Meaning-Form Correlation Has to Compose With
This is the repository for the [COLING 2020](https://coling2020.org/) paper ["What Meaning-Form Correlation Has to Compose With"](nowhere.org/place-holder).

Much assembly required. This repository is a hot mess of monkeypatches held together by off-brand brown tape.

## Install, external dependencies, acknowledgments

Most dependencies are covered in the `pip.requirements.txt` file. You might need to install a spacy model for English:
````
python3 -m spacy download en_core_web_sm
````
You may also need to download NLTK-Wordnet:
````
python3 -c "import nltk; nltk.download('wordnet')"
````

Mantel tests are from J. W. Carr's github ([see here](https://github.com/jwcarr/MantelTest)). We only include it here for convenience, the original code hasn't been modified.

The implementation of APTED by Pawlik & Augsten is from the [repository](https://github.com/DatabaseGroup/apted) (written in Java). The JAR we provide has been hacked to accept a file of pairs of trees at once, instead of a single pair: calls should resemble something like:
````
java -jar apted.jar trees.tsv
````
The single edited Java file is available for reference in the directory `shared/apted/`.

Pre-trained word embeddings are available from their original repositories, or at these links: [Word2Vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing), [GloVe 6B](http://nlp.stanford.edu/data/glove.6B.zip) and [840B](http://nlp.stanford.edu/data/glove.840B.300d.zip), [FastText](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip)

Sentence encoders are from the original repositories: [SkipThoughts](https://github.com/ryankiros/skip-thoughts) (written in python 2), [InferSent](https://github.com/facebookresearch/InferSent).
See also the original Google Hub for [USE DAN](https://tfhub.dev/google/universal-sentence-encoder/4) and [USE Transformer](https://tfhub.dev/google/universal-sentence-encoder-large/5).

Lastly, the two datasets used to evaluate embeddings, the [MEN](https://staff.fnwi.uva.nl/e.bruni/MEN) by Bruni et al. and [SICK](http://marcobaroni.org/composes/sick.html) by Baroni et al. can be retrieved from their respective homepages.

## Structure

Code is stored in `src/`. The `src/shared/` directory contains pieces of code shared by some or all experiments. Script starting with `src/exp1_`, `src/exp2_` and `src/exp3_` correspond to code for artificial language experiments, definition experiments and sentence experiments respectively. Subdirectory `src/exp3_embs` contains specifically scripts to compute or retrieve sentence embeddings.

Data is available under `data/`, subdirectories correspond to different experiments.
