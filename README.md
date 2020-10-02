# mf-correl
Repository for the paper ["What Meaning-Form Correlation Has to Compose With"](nowhere.org).

Much assembly required. This repository is a hot mess of monkeypatches held together by off-brand brown tape.

## Install, external dependencies, aknowledgments

Most dependencies are covered in the `pip.requirements.txt` file.

Mantel tests are from J. W. Carr's github ([see here](https://github.com/jwcarr/MantelTest))

The implementation of APTED by Pawlik & Augsten is from the [repository](https://github.com/DatabaseGroup/apted) (written in Java). The JAR we provide has been hacked to accept a file of pairs of trees at once, instead of a single pair.

Pre-trained ord embeddings are available from their original repositories, links: [Word2Vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing), [GloVe 6B](http://nlp.stanford.edu/data/glove.6B.zip) and [840B](http://nlp.stanford.edu/data/glove.840B.300d.zip), [FastText](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip)

Sentence encoders are from the original repositories: [SkipThoughts](https://github.com/ryankiros/skip-thoughts) (written in python 2), [InferSent](https://github.com/facebookresearch/InferSent).
See also the original Google Collab for [USE DAN](https://tfhub.dev/google/universal-sentence-encoder/4) and [USE Transformer](https://tfhub.dev/google/universal-sentence-encoder-large/5).

