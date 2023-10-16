import numpy as np
import pandas as pd
import os

import sklearn.linear_model
import sklearn.pipeline

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle


def main():
    x_train_df, y_train_df = load_train_data()

    # getting rid of company label
    x_train_N = x_train_df.to_numpy()[:, 1]
    y_train_N = y_train_df.to_numpy()[:, 0]

    # shuffle dataset
    x_train_N, y_train_N = shuffle(x_train_N, y_train_N, random_state=2)

    # TODO: add in GridSearchCV with cross validation and hyperparams for logistic regression

    # print_word_freq(x_train_N)

    pipeline = make_bow_classifier_pipeline()
    pipeline.fit(x_train_N, y_train_N)

    yhat_train_N = pipeline.predict(x_train_N)

    accuracy = np.mean(yhat_train_N == y_train_N)

    print(f"training data accuracy: {accuracy}")


# imports training data from files
def load_train_data():
    data_dir = "data_reviews"
    x_train_df = pd.read_csv(os.path.join(data_dir, "x_train.csv"))
    y_train_df = pd.read_csv(os.path.join(data_dir, "y_train.csv"))
    return x_train_df, y_train_df


def print_word_freq(x_train_N):
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


# pipeline for BoW representation + logistic regression classifier
# TODO: add params for vectorizer and classifier
def make_bow_classifier_pipeline():
    pipeline = sklearn.pipeline.Pipeline(
        steps=[
            # turn data into BoW feature representation
            (
                "bow_feature_extractor",
                CountVectorizer(min_df=1, max_df=1.0, ngram_range=(1, 1)),
            ),
            # TODO: add cross validation
            # Given features construct the classifier (w/ hyperparam selection)
            (
                "classifier",
                sklearn.linear_model.LogisticRegression(C=1.0),
            ),
        ]
    )

    return pipeline


if __name__ == "__main__":
    main()
