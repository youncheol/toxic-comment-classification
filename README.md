# Toxic Comment Classification

[*Identify and classify toxic online comments*](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

wikipedia talk의 코멘트 데이터에서 악성 코멘트를 골라내고, 그것들이 6가지 악성요소(`toxic`, `severe_toxic`, `threats`, `obscene`, `insult`, `identity_hate`) 중 어떤 특성을 띄는지 분류 (multi-label 문제)



## Requirements

모델 학습을 위해 미리 학습된 [GloVe](https://nlp.stanford.edu/projects/glove) 벡터를 gensim에서 사용가능한 포맷으로 변경

Example:

```
$ python convert_glove_format.py --source_fname "/your_path/glove.twitter.27B.100d.txt"
```



## Model

* [A Convolutional Attention Model for Text Classification](http://tcci.ccf.org.cn/conference/2017/papers/1057.pdf) 모델을 텐서플로우로 구현

* LSTM cell을 GRU cell로 교체하고 loss function을 sigmoid cross entropy로 변경



## Preprocessing

Example:

```
$ python preprocessing.py --dafa_fname "train.csv"
```

Optional arguments:

```
$ python preprocessing.py -h

optional arguments:
  -h, --help            show this help message and exit
  --data_fname DATA_FNAME
                        file name of data to processing
  --tfr_fname TFR_FNAME
                        file name of TFRecord to be created (default: train.tfrecord)
  --glove_fname GLOVE_FNAME
                        file name of pre-trained GloVe embedding model
                        (default: glove.model)
  --max_length MAX_LENGTH
                        threshold length to truncate comments (default: 300)
  --train               use training data (default: False)
```



## Train

Example:

```
$ python train.py
```

Optional arguments:

```
$ python train.py -h

optional arguments:
  -h, --help            show this help message and exit
  --glove_fname GLOVE_FNAME
                        file name of pre-trained GloVe embedding model
                        (default: glove.model)
  --tfr_fname TFR_FNAME
                        file name of TFRecord to train (default: train.tfrecord)
  --num_epochs NUM_EPOCHS
                        number of training epochs (default: 5)
  --logdir LOGDIR       directory name where to write log files (default: logs)
  --save_fname SAVE_FNAME
                        prefix of model to be saved (default: model
  --batch_size BATCH_SIZE
                        batch size to load data (default: 64)
  --filter_size FILTER_SIZE
                        size of convolution filters (default: 3)
  --num_filters NUM_FILTERS
                        number of convolution channels (default: 100)
  --hidden_size HIDDEN_SIZE
                        number of GRU hidden units (default: 100)
  --learning_rate LEARNING_RATE
                        learning rate for optimization (default: 0.001)
  --dropout_prob DROPOUT_PROB
                        dropout probability of RNN layer (default: 0.5)
  --class_weight CLASS_WEIGHT [CLASS_WEIGHT ...]
                        class weight of loss function (default: None)
```



## Predict

Example:

```
$ python predict.py --model_fname "your_model"
```

Optional arguments:

```
$ python predict.py -h

optional arguments:
  -h, --help            show this help message and exit
  --model_fname MODEL_FNAME
                        file name of model to restore
  --glove_fname GLOVE_FNAME
                        file name of pre-trained GloVe embedding model
                        (default: glove.model)
  --tfr_fname TFR_FNAME
                        file name of TFRecord to predict (default: test.tfrecord)
  --sample_fname SAMPLE_FNAME
                        file name of kaggle sample submission (default:
                        sample_submission.csv)
  --output_fname OUTPUT_FNAME
                        file name of submission to be created (default: submission.csv)
  --batch_size BATCH_SIZE
                        batch size to load data (default: 64)
  --filter_size FILTER_SIZE
                        size of convolution filters (default: 3)
  --num_filters NUM_FILTERS
                        number of convolution channels (default: 100)
  --hidden_size HIDDEN_SIZE
                        number of GRU hidden units (default: 100)
  --learning_rate LEARNING_RATE
                        learning rate for optimization (default: 0.001)
  --dropout_prob DROPOUT_PROB
                        dropout probability of RNN layer (default: 0.5)
  --proba               predict probabilities (default: False)
```



## Result

* AUC score: 0.9813