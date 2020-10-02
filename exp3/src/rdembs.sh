
echo python3 get_rand_embs.py --input_dir sentences/raw_data/ --pickle randembs.pkl;
python3 get_rand_embs.py --input_dir sentences/raw_data/ --pickle randembs.pkl;

echo mkdir -p sentences/embs/rd-embs/;
mkdir -p sentences/embs/rd-embs/;

echo mv sentences/raw_data/*.tsv sentences/embs/rd-embs/;
mv sentences/raw_data/*.tsv sentences/embs/rd-embs/;

for file in $(find sentences/embs/rd-embs/ -type f -name '*.tsv'); do 
	echo python3 compotest_sentences.py --input ${file} --output ${file}.json;
	python3 compotest_sentences.py --input ${file} --output ${file}.json; 
done

echo python3 compute_mantels_per_dir.py --input_dir sentences/embs/rd-embs/ --output mantels_rd-embs.tsv;
python3 compute_mantels_per_dir.py --input_dir sentences/embs/rd-embs/ --output mantels_rd-embs.tsv;

echo sort -k1,3 -o mantels_rd-embs.tsv mantels_rd-embs.tsv;
sort -k1,3 -o mantels_rd-embs.tsv mantels_rd-embs.tsv;

echo python3 test_sick.py --emb_arch randembs --emb_path randembs.pkl;
python3 test_sick.py --emb_arch randembs --emb_path randembs.pkl;
