# Name: Graeme Hosford
# Student ID: R00147327

import numpy as np
import pandas as pd
import re

from itertools import chain
from collections import Counter

data = pd.read_excel("movie_reviews.xlsx")


def split_data():
    is_train_data = data["Split"] == "train"
    is_test_data = data["Split"] == "test"

    review_training_data = data["Review"][is_train_data]
    labels_training_data = data["Sentiment"][is_train_data]

    review_test_data = data["Review"][is_test_data]
    labels_test_data = data["Sentiment"][is_test_data]

    print("\nCount of positive and negative reviews in the training data set:")
    print(labels_training_data.value_counts())

    print("\nCount of positive and negative reviews in the testing data set:")
    print(labels_test_data.value_counts())

    return review_training_data, labels_training_data, review_test_data, labels_test_data


def count_training_words(training_reviews: pd.Series, min_word_length, min_word_occurance):
    training_reviews.replace(to_replace="[^A-Za-z0-9\s]+", value="", regex=True, inplace=True)
    training_reviews = training_reviews.apply(lambda x: x.lower())

    word_freq_dict = dict()

    for review in training_reviews:
        words = review.lower().split()
        for word in words:
            if len(word) >= min_word_length:
                if word in word_freq_dict.keys():
                    word_freq_dict[word] += 1
                else:
                    word_freq_dict[word] = 0

    new_dict = dict()

    for key, value in word_freq_dict.items():
        if value >= min_word_occurance:
            new_dict[key] = value

    return training_reviews, list(new_dict.keys())


def count_word_in_review_frequency(review_list,  word_list):
    word_freq_dict = dict()

    for review in review_list:
        for word in review.split():
            if word in word_list:
                if word in word_freq_dict.keys():
                    word_freq_dict[word] += 1
                else:
                    word_freq_dict[word] = 0

    return word_freq_dict


def main():
    review_train_data, label_train_data, review_test_data, label_test_data = split_data()
    review_train_data, word_list = count_training_words(review_train_data, 3, 400)

    word_set = set(word_list)

    positive_training_reviews = review_train_data[label_train_data == "positive"]
    negative_training_reviews = review_train_data[label_train_data == "negative"]

    positive_word_freq_dict = count_word_in_review_frequency(positive_training_reviews, word_set)
    negative_word_freq_dict = count_word_in_review_frequency(negative_training_reviews, word_set)


main()
