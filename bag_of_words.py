import numpy as np
import pandas as pd
import os

import sklearn.linear_model
import sklearn.pipeline

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle


def main():
    x_train_df, y_train_df = load_train_data()

    x_test_df = load_test_data()

    # getting rid of company label
    x_train_N = x_train_df.to_numpy()[:, 1]
    y_train_N = y_train_df.to_numpy()[:, 0]

    x_test_N = x_test_df.to_numpy()[:, 1]

    # shuffle dataset
    x_train_N, y_train_N = shuffle(x_train_N, y_train_N, random_state=2)

    pipeline = make_bow_classifier_pipeline()
    hyperparam_grid = make_hyperparam_grid()

    grid_searcher = sklearn.model_selection.GridSearchCV(
        pipeline,
        hyperparam_grid,
        cv=10,
        scoring="roc_auc"
    )

    grid_searcher.fit(x_train_N, y_train_N)

    # grid_search_results_df = pd.DataFrame(grid_searcher.cv_results_).copy()

    # param_keys = ['param_bow_feature_extractor__min_df', 'param_bow_feature_extractor__max_df', 'param_classifier__C']

    # grid_search_results_df.sort_values(param_keys, inplace=True)
    # grid_search_results_df[param_keys + ['split0_test_score', 'rank_test_score']]

    # print(grid_search_results_df)

    best_model = grid_searcher.best_estimator_
    best_params = grid_searcher.best_params_
    best_score = grid_searcher.best_score_

    yhat_proba_test_N = best_model.predict_proba(x_test_N)

    np.savetxt("yproba1_test.txt", yhat_proba_test_N[:,1])

    print(f"best score: {best_score}; best params:", best_params)





# imports training data from files
def load_train_data():
    data_dir = "data_reviews"
    x_train_df = pd.read_csv(os.path.join(data_dir, "x_train.csv"))
    y_train_df = pd.read_csv(os.path.join(data_dir, "y_train.csv"))
    return x_train_df, y_train_df

def load_test_data():
    data_dir = "data_reviews"
    x_test_df = pd.read_csv(os.path.join(data_dir, "x_test.csv"))
    return x_test_df

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
                CountVectorizer(min_df=2, max_df=1.0, ngram_range=(1, 1)),
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


def make_hyperparam_grid():

    hyperparam_grid = {}

    hyperparam_grid["bow_feature_extractor__min_df"] = [1, 2, 3, 4]
    hyperparam_grid["bow_feature_extractor__max_df"] = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    hyperparam_grid["classifier__C"] = np.logspace(-1, 1, 11)

    return hyperparam_grid


if __name__ == "__main__":
    main()