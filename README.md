# Transfer Learning for Neural Topic Models using Knowledge Distillation

# Rough Steps

## 1. Create conda environment (which is a modification of [Scholar+BAT](https://github.com/ahoho/kd-topic-models))
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

- Fine-tuning the neural topic model with the target data using the obtained parameters.
```
python scholar/init_embeddings.py data/imdb/processed-dev/train.vocab.json  --teacher-vocab data/wiki20200501/processed-dev/train.vocab.json  --model-file outputs/wiki_topic_500_emb_dim_500/torch_model_epoch_100.pt  -o ./scholar/outputs/imdb/wiki_topic_500_emb_500_epoch_100
```
```
python scholar/run_scholar.py 
./data/imdb/processed-dev 
--dev-metric npmi -k 50  
--epochs 500  
--patience 50 
--batch-size 200 
--background-embeddings scholar/outputs/imdb/wiki_topic_500_emb_500_epoch_100/pretrained_emb.npy  
--device 0 
--dev-prefix test 
-l 0.002 
--alpha 0.5 
--eta-bn-anneal-step-const 0.25 
-o ./outputs/teacher_imdb/test/topic_500_emb_500_epoch_100
--save-eta  
```

## 4. Konwledge Distillation 
```
python scholar/run_scholar.py
./data/imdb/processed-dev
--dev-metric npmi 
-k 50 
--epochs 500 
--patience 50 
--batch-size 200 
--background-embeddings 
--device 0 
--dev-prefix test 
-l 0.002 
--alpha 0.5 
--eta-bn-anneal-step-const 0.25 
--teacher-eta-dirs ./outputs/teacher_imdb/test/topic_500_emb_500_epoch_100
--no-bow-reconstruction-loss 
--doc-reconstruction-weight-list 0.97
--doc-reconstruction-temp-list 2.0
-o ./outputs/kd_imdb/test/topic_500_emb_500_epoch_100
--use-pseudo-doc
```

