import numpy as np
import pandas as pd
import os

from sklearn.feature_extraction.text import CountVectorizer


def main():
    x_train_df, y_train_df = load_train_data()

    # getting rid of company label
    x_train_N = x_train_df.to_numpy()[:, 1]

    # min_df sets sets a minimum number of times a given token needs to be
    # included in a text entry to be a part of the vector
    vectorizer = CountVectorizer(min_df=4, binary=False)
    x_vec_N = vectorizer.fit_transform(x_train_N)

    dense_arr = x_vec_N.toarray()

    freq = [
        (term, np.sum(dense_arr[:, index]))
        for term, index in sorted(list(vectorizer.vocabulary_.items()))
    ]

    for term, count in sorted(freq, key=lambda x: x[1], reverse=True):
        print(f"{term} -- {count}")


def load_train_data():
    data_dir = "data_reviews"
    x_train_df = pd.read_csv(os.path.join(data_dir, "x_train.csv"))
    y_train_df = pd.read_csv(os.path.join(data_dir, "y_train.csv"))
    return x_train_df, y_train_df


if __name__ == "__main__":
    main()
