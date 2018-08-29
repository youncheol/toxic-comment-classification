import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
from data_processor import PreProcessing, TFRecord
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_fname", required=True,
                        help="file name of data to processing")
    parser.add_argument("--tfr_fname", default="train.tfrecord",
                        help="file name of TFRecord to be created (default: train.tfrecord)")
    parser.add_argument("--glove_fname", default="glove.model",
                        help="file name of pre-trained GloVe embedding model (default: glove.model)")
    parser.add_argument("--max_length", default=300, type=int,
                        help="threshold length to truncate comments (default: 300)")
    parser.add_argument("--train", default=False, action="store_true",
                        help="use training data (default: False)")

    return parser.parse_args()


def main():
    args = get_args()

    df = pd.read_csv(args.data_fname)

    prep = PreProcessing()
    tfrecord = TFRecord()

    prep.processing(df, args.glove_fname, args.max_length)

    if args.train:
        train_y = df.as_matrix()[:, 2:].astype("int32")
        tfrecord.save(args.tfr_fname, prep.indexed_data, train_y, training=True)

    else:
        tfrecord.save(args.tfr_fname, prep.indexed_data, training=False)


if __name__ == "__main__":
    main()
