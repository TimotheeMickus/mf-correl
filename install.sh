#!/usr/bin/env bash

echo '0. Setting up virtual environments'
echo '0a. Setting up python3 environment'
virtualenv --python=/usr/bin/python3.7 .venv3
source .venv3/bin/activate
pip3 install -r pip3.requirements.txt
python3 -m spacy download en_core_web_sm
python3 -c "import nltk; nltk.download('wordnet')"
python3 -c "import nltk; nltk.download('punkt')"
deactivate

echo '0a. Setting up python2 environment'
virtualenv --python=/usr/bin/python2 .venv2
source .venv2/bin/activate
pip2 install -r pip2.requirements.txt
deactivate


echo '1. Retrieving word embeddings'
for ARCH in w2v ft gv6 gv840
do
  mkdir -p data/exp2/embs/${ARCH}
done
echo '1a. Retrieving word2vec (GoogleNews)'
source .venv3/bin/activate
gdown https://drive.google.com/u/0/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM;
gunzip -d GoogleNews-vectors-negative300.bin.gz
mv GoogleNews-vectors-negative300.bin data/exp2/embs/w2v/
deactivate;

echo '1b. Retrieving fastText (CommonCrawl)'
wget -P data/exp2/embs/ft/ https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
gunzip -d data/exp2/embs/ft/cc.en.300.bin.gz

echo '1c. Retrieving GloVe (6B)'
wget -P data/exp2/embs/gv6/ http://nlp.stanford.edu/data/glove.6B.zip
unzip -d data/exp2/embs/gv6/ data/exp2/embs/gv6/glove.6B.zip

echo '1d. Retrieving GloVe (840B)'
wget -P data/exp2/embs/gv840/ http://nlp.stanford.edu/data/glove.840B.300d.zip;
unzip -d data/exp2/embs/gv840/ data/exp2/embs/gv840/glove.840B.300d.zip


echo '2. Download & set up external git repos'
echo "2a. J. W. Carr's MantelTest"
git clone https://github.com/jwcarr/MantelTest.git src/shared/MantelTest

echo "2b. R. Kyros Skip-Thoughts"
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
sed -i "s#'/u/rkiros/public_html/models/'#os.path.join(os.path.dirname(__file__), 'models/')#g" skipthoughts.py
cd ../../..

echo "2c. FB's InferSent"
git clone https://github.com/facebookresearch/InferSent.git src/exp3_embs/InferSent
ln -s data/exp2/embs/gv840 src/exp3_embs/InferSent/embs
mkdir src/exp3_embs/InferSent/encoder
wget -P src/exp3_embs/InferSent/encoder https://dl.fbaipublicfiles.com/infersent/infersent1.pkl


echo '3. Evaluation data'
echo '3a. Bruni & al. MEN dataset'
wget -P data/ https://staff.fnwi.uva.nl/e.bruni/resources/MEN.zip
unzip -d data data/MEN.zip

echo '3b. Baroni & al. SICK dataset'
wget -P data/ https://zenodo.org/record/2787612/files/SICK.zip
unzip -d data/SICK data/SICK.zip
