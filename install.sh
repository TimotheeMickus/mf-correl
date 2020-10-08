#!/usr/bin/env bash

echo '1a. Setting up python3 environment'
virtualenv --python=/usr/bin/python3.7 .venv3
source .venv3/bin/activate
pip3 install -r pip3.requirements.txt

python3 -m spacy download en_core_web_sm
python3 -c "import nltk; nltk.download('wordnet')"
python3 -c "import nltk; nltk.download('punkt')"

deactivate;

echo '1a. Setting up python2 environment'
virtualenv --python=/usr/bin/python2 .venv2
source .venv2/bin/activate
pip2 install -r pip2.requirements.txt

deactivate;

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
sed -i 's#/u/rkiros/public_html/models/#os.path.join(os.path.dirname(__file__), "models/")#g' skipthoughts.py
cd ../../..

echo "2c. FB's InferSent"
git clone https://github.com/facebookresearch/InferSent.git src/exp3_embs/InferSent
cd src/exp3_embs/InferSent;
mkdir embs
wget -P embs/ http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip embs/glove.840B.300d.zip -d embs/
mkdir encoder
wget -P encoder https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
cd ../../..

echo '3. Evaluation data'
wget -P data/ https://staff.fnwi.uva.nl/e.bruni/resources/MEN.zip
unzip -d data data/MEN.zip
wget -P data/ https://zenodo.org/record/2787612/files/SICK.zip
unzip -d data/SICK data/SICK.zip
