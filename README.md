# What Meaning-Form Correlation Has to Compose With
This is the repository for the [COLING 2020](https://coling2020.org/) paper ["What Meaning-Form Correlation Has to Compose With"](nowhere.org/place-holder).

Much assembly required. This repository is a hot mess of monkeypatches held together by off-brand brown tape.

## Install, external dependencies, acknowledgments

Code has been tested and developed with python 3.7 (Ubuntu 18.04). Most dependencies are covered in the `pip3.requirements.txt` file; some might require time to install (e.g., Tensorflow 1 only ships with python < 3.8 for some reason; Spacy 2.1.9 seems to now have broken down in flames due to an issue in one of its dependencies). You might need to install a spacy model for English:
````{.sh}
python3 -m spacy download en_core_web_sm
````
You may also need to download NLTK-Wordnet, as well as punkt for infersent:
````{.sh}
python3 -c "import nltk; nltk.download('wordnet')"
python3 -c "import nltk; nltk.download('punkt')"
````

We used Mantel tests are from J. W. Carr's github ([see here](https://github.com/jwcarr/MantelTest)).
Install at the expected location with:
````{.sh}
git clone https://github.com/jwcarr/MantelTest.git src/shared/MantelTest
````

The implementation of APTED by Pawlik & Augsten is from the [repository](https://github.com/DatabaseGroup/apted) (written in Java). The JAR we provide has been hacked to accept a file of pairs of trees at once, instead of a single pair: calls should resemble something like:
````{.sh}
java -jar apted.jar trees.tsv
````
The single edited Java file is available for reference in the directory `shared/apted/`.

Pre-trained word embeddings are available from their original repositories, or at these links: [Word2Vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing), [GloVe 6B](http://nlp.stanford.edu/data/glove.6B.zip) and [840B](http://nlp.stanford.edu/data/glove.840B.300d.zip), [FastText](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip)

Sentence encoders are from the original repositories: [SkipThoughts](https://github.com/ryankiros/skip-thoughts) (written in python 2), [InferSent](https://github.com/facebookresearch/InferSent).
See also the original Google Hub for [USE DAN](https://tfhub.dev/google/universal-sentence-encoder/4) and [USE Transformer](https://tfhub.dev/google/universal-sentence-encoder-large/5).

Setting up skip-thoughts requires some steps:
````{.sh}
git clone https://github.com/ryankiros/skip-thoughts.git src/exp3_embs/skip_thoughts
cd src/exp3_embs/skip_thoughts; mkdir models; cd models
wget http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
wget http://www.cs.toronto.edu/~rkiros/models/utable.npy
wget http://www.cs.toronto.edu/~rkiros/models/btable.npy
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl
cd ..
sed -i 's#/u/rkiros/public_html/models/#os.path.join(os.path.dirname(__file__), "models/")#g' skipthoughts.py
cd ../../..
````
This should download the original github under `src/exp3_embs/skip_thoughts/`, download the required theano model files and update the skipthoughts file coherently. Now you only need to create a virtual environment for python2 and install the pip dependencies listed in `pip2.requirements.txt`.

InferSent can be set up with the following:
````{.sh}
git clone https://github.com/facebookresearch/InferSent.git src/exp3_embs/InferSent
cd src/exp3_embs/InferSent;
mkdir embs
wget -P embs/ http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip embs/glove.840B.300d.zip -d embs/
mkdir encoder
wget -P encoder https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
cd ../../..
````

Lastly, the two datasets used to evaluate embeddings, the [MEN](https://staff.fnwi.uva.nl/e.bruni/MEN) by Bruni et al. and [SICK](http://marcobaroni.org/composes/sick.html) by Baroni et al. can be retrieved from their respective homepages.

## Structure

Code is stored in `src/`. The `src/shared/` directory contains pieces of code shared by some or all experiments. Script starting with `src/exp1_`, `src/exp2_` and `src/exp3_` correspond to code for artificial language experiments, definition experiments and sentence experiments respectively. Subdirectory `src/exp3_embs/` contains specifically scripts to compute or retrieve sentence embeddings.

Data is available under `data/`, subdirectories correspond to different experiments.
