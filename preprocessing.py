import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
from data_processor import PreProcessing, TFRecord
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--data_fname", required=True)
    parser.add_argument("--tfr_fname", required=True)
    parser.add_argument("--glove_fname", required=True)
    parser.add_argument("--max_length", type=int)
    args = parser.parse_args()

    df = pd.read_csv(args.data_fname)

    prep = PreProcessing()
    tfrecord = TFRecord()

    prep.processing(df, args.glove_fname, args.max_length)

    if args.train:
        train_y = df.as_matrix()[:, 2:].astype('int32')
        tfrecord.save(args.tfr_fname, prep.indexed_data, train_y, training=True)

    else:
        tfrecord.save(args.tfr_fname, prep.indexed_data, training=False)

if __name__ == "__main__":
    main()