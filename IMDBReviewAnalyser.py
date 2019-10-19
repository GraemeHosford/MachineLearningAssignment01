# Name: Graeme Hosford
# Student ID: R00147327

from collections import Counter

import pandas as pd


def split_data():
    data = pd.read_excel("movie_reviews.xlsx")

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


def clean_data(review_data: pd.Series) -> pd.Series:
    review_data.replace(to_replace="[^A-Za-z0-9\s]+", value="", regex=True, inplace=True)
    return review_data.apply(lambda x: x.lower())


def count_training_words(training_reviews: pd.Series, min_word_length, min_word_occurance) -> set:
    word_freq_dict = dict()

    for review in training_reviews:
        for word in review.split():
            if len(word) >= min_word_length:
                if word in word_freq_dict.keys():
                    word_freq_dict[word] += 1
                else:
                    word_freq_dict[word] = 1

    new_dict = dict()

    for key, value in word_freq_dict.items():
        if value >= min_word_occurance:
            new_dict[key] = value

    return set(new_dict.keys())


def count_word_in_review_frequency(review_list: pd.Series, word_set: set) -> Counter:
    word_freq_dict = Counter(word_set)

    for k in word_freq_dict.keys():
        word_freq_dict[k] = 0

    for review in review_list:
        for word in word_set:
            if word in review:
                word_freq_dict[word] += 1

    return word_freq_dict


def calculate_likelihood(word_freq: Counter, data_set: pd.Series) -> dict:
    word_likelihood = dict()
    smoothing = 1

    for word in word_freq.keys():
        word_likelihood[word] = (word_freq[word] + smoothing) / (len(data_set) + smoothing)

    return word_likelihood


def calculate_priors(full_dataset: pd.Series, subset: pd.Series):
    return len(subset) / len(full_dataset)


def main():
    review_train_data, label_train_data, review_test_data, label_test_data = split_data()

    review_train_data = clean_data(review_train_data)

    word_set = count_training_words(review_train_data, 4, 100)

    positive_training_reviews = review_train_data[label_train_data == "positive"]
    negative_training_reviews = review_train_data[label_train_data == "negative"]

    positive_word_freq_dict = count_word_in_review_frequency(positive_training_reviews, word_set)
    negative_word_freq_dict = count_word_in_review_frequency(negative_training_reviews, word_set)

    positive_likelihood = calculate_likelihood(positive_word_freq_dict, positive_training_reviews)
    negative_likelihood = calculate_likelihood(negative_word_freq_dict, negative_training_reviews)

    positive_prior = calculate_priors(review_train_data, positive_training_reviews)
    negative_prior = calculate_priors(review_train_data, negative_training_reviews)

    print(positive_prior)
    print(negative_prior)


main()
