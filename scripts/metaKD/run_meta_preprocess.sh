# 1. Download Amazon dataset
mkdir data
cd data
wget https://www.cs.jhu.edu/~mdredze/datasets/sentiment/domain_sentiment_data.tar.gz
tar -zxvf domain_sentiment_data.tar.gz
cd ..

# 2. Split the dataset
python generate_senti_data.py

# 3. extract bert embedding
python extract_embeddings.py \
  --bert_path ~/tools/bert/bert-base-uncased \
  --input data/SENTI/train.tsv \
  --output data/SENTI/train.embeddings.tsv \
  --task_name senti --gpu 2

# 4. generate instance weights
python generate_meta_weights.py \
  data/SENTI/train.embeddings.tsv \
  data/SENTI/train_with_weights.tsv \
  books,dvd,electronics,kitchen