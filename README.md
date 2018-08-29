# Toxic Comment Classification

## Requirements

모델 학습을 위해서 먼저 미리 학습된 [GloVe](https://nlp.stanford.edu/projects/glove) 벡터를 gensim에서 사용가능한 포맷으로 변경해줘야 합니다. 

실행 예:

```
$ python convert_glove_format.py --source_fname "/yourpath/glove.twitter.27B.100d.txt" --target_fname "glove"
```



## Train

실행 예:

```
$ python train.py --glove_fname "glove.model" --tfr_fname "train.tfrecord" --num_epochs 5 --logdir "log_path" --save_fname "model_name"
```

optional parameters:

```
$ python train.py -h

optional arguments:
  -h, --help            show this help message and exit
  --glove_fname GLOVE_FNAME
                        file name of pre-trained GloVe embedding model
  --tfr_fname TFR_FNAME
                        file name of TFRecord to train
  --num_epochs NUM_EPOCHS
                        number of training epochs
  --logdir LOGDIR       directory name where to write log files
  --save_fname SAVE_FNAME
                        prefix of model to be saved
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

```
python predict.py -h

optional arguments:
  -h, --help            show this help message and exit
  --glove_fname GLOVE_FNAME
                        file name of pre-trained GloVe embedding model
  --model_fname MODEL_FNAME
                        file name of model to restore
  --tfr_fname TFR_FNAME
                        file name of TFRecord to predict
  --sample_fname SAMPLE_FNAME
                        file name of kaggle sample submission
  --output_fname OUTPUT_FNAME
                        file name of submission to be created
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

