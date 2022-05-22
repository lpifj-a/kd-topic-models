# Transfer Learning for Neural Topic Models using Knowledge Distillation

This repository is a modified version of kd-topic for this study

# Rough Steps

## 1. Create conda environment (which is a modification of [Scholar](https://github.com/dallascard/scholar)).
```
conda env create -f scholar/scholar.yml
```
## 2. Preprocess the data

### Student data
- download the data
```
python data/imdb/download_imdb.py
```
- main preprocessing script
```
python preprocess_data.py ~/kd-topic-models/data/imdb/train.jsonlist ~/kd-topic-models/data/imdb/processed --vocab-size 5000 --test ~/kd-topic-models/data/imdb/test.jsonlist --label rating
```
- create a dev split from the train data
```
python data/imdb/create_dev_split.py
```

### Teacher data
- download the data
```
python data/wiki20200501/download_wiki.py
```
- main preprocessing script
```
python preprocess_data.py ~/kd-topic-models/data/wiki20200501/train.jsonlist ~/kd-topic-models/data/wiki20200501/processed --vocab-size 50000
```
- create a dev split from the train data
```
python data/wiki20200501/create_dev_split.py
```

## 3. Run the teacher model
- Pre-training the parameters of the inference network of the neural topic model using the Wikipedia dataset
```
python scholar/run_scholar.py ./data/wiki20200501/processed-dev  -k 500 --emb-dim 500  --epochs 500 --batch-size 5000 --background-embeddings --device 0 -l 0.002 --alpha 0.5 --eta-bn-anneal-step-const 0.25 -o ./outputs/wiki_topic_500_emb_dim_500  --save-for-each-epoch 10 
```

- Fine-tune the neural topic model with the target data using the obtained parameters.
```
python scholar/init_embeddings.py data/imdb/processed-dev/train.vocab.json  --teacher-vocab data/wiki20200501/processed-dev/train.vocab.json  --model-file outputs/wiki_topic_500_emb_dim_500/torch_model_epoch_100.pt  -o ./scholar/outputs/imdb/wiki_topic_500_emb_500_epoch_100
```
```
python multiple_run_scholar.py $(cat args/imdb/scholar_wiki.txt)
```

## 4. Konwledge Distillation 
```
python multiple_run_scholar.py $(cat args/imdb/wiki_kd.txt)
```

