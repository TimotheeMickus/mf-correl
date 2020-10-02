
echo python3 get_randtf_embs.py --input_dir sentences/raw_data/ --pickle randtf-2.pkl;
python3 get_randtf_embs.py --input_dir sentences/raw_data/ --pickle randtf-2.pkl;

echo mv sentences/raw_data/*.tsv sentences/embs/rd-tf-2/;
mv sentences/raw_data/*.tsv sentences/embs/rd-tf-2/;

for file in $(find sentences/embs/rd-tf-2/ -type f -name '*.tsv'); do 
	echo python3 compotest_sentences.py --input ${file} --output ${file}.json;
	python3 compotest_sentences.py --input ${file} --output ${file}.json; 
done

echo python3 compute_mantels_per_dir.py --input_dir sentences/embs/rd-tf-2/ --output mantels_rd-tf-2.tsv;
python3 compute_mantels_per_dir.py --input_dir sentences/embs/rd-tf-2/ --output mantels_rd-tf-2.tsv;

echo sort -k1,3 -o mantels_rd-tf-2.tsv mantels_rd-tf-2.tsv;
sort -k1,3 -o mantels_rd-tf-2.tsv mantels_rd-tf-2.tsv;

echo python3 test_sick.py --emb_arch randtf --emb_path randtf-2.pkl;
python3 test_sick.py --emb_arch randtf --emb_path randtf-2.pkl;
