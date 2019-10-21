# Name: Graeme Hosford
# Student ID: R00147327

import math
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
    review_data.replace(to_replace="[^A-Za-z0-9\\s]+", value="", regex=True, inplace=True)
    return review_data.apply(lambda x: x.lower())


def count_training_words(training_reviews: pd.Series, min_word_length, min_word_occurance) -> set:
    word_freq_dict = Counter()

    for index, review in training_reviews.iteritems():
        words = set(filter(lambda x: len(x) >= min_word_length, review.split()))
        word_freq_dict.update(words)

    words = set()

    for key, value in word_freq_dict.items():
        if value >= min_word_occurance:
            words.add(key)

    return words


def count_word_in_review_frequency(review_list: pd.Series, word_set: set) -> Counter:
    word_freq_dict = Counter(word_set)

    for k in word_freq_dict.keys():
        word_freq_dict[k] = 0

    for review in review_list:
        words = set(review.split())
        word_freq_dict.update(set.intersection(words, word_set))

    return word_freq_dict


def calculate_likelihood(word_freq: Counter) -> dict:
    word_likelihood = dict()
    smoothing = 1

    total = sum(word_freq.values())

    for word in word_freq.keys():
        word_likelihood[word] = (word_freq[word] + smoothing) / (total + (len(word_freq) + smoothing))

    return word_likelihood


def calculate_priors(full_dataset: pd.Series, subset: pd.Series) -> float:
    return len(subset) / len(full_dataset)


def classify_text(review_text: str, positive_likelihood: dict, negative_likelihood: dict,
                  positive_prior: float, negative_prior: float) -> bool:
    new_text_words = review_text.lower().replace("[^A-Za-z0-9\s]+", "").split()

    positive_log = 0

    for word in new_text_words:
        if positive_likelihood.get(word, 0) > 0:
            positive_log += math.log(positive_likelihood.get(word, 0))

    negative_log = 0

    for word in new_text_words:
        if negative_likelihood.get(word, 0) > 0:
            negative_log += math.log(negative_likelihood.get(word, 0))

    likelihood_ratio = math.exp(positive_log - negative_log)
    prior_ratio = math.exp(math.log(negative_prior) - math.log(positive_prior))

    return likelihood_ratio > prior_ratio


def main():
    review_train_data, label_train_data, review_test_data, label_test_data = split_data()

    review_train_data = clean_data(review_train_data)

    word_set = count_training_words(review_train_data, 1, 1)

    positive_training_reviews = review_train_data[label_train_data == "positive"]
    negative_training_reviews = review_train_data[label_train_data == "negative"]
    positive_test_reviews = review_test_data[label_test_data == "positive"]
    negative_test_reviews = review_test_data[label_test_data == "negative"]

    positive_word_freq_dict = count_word_in_review_frequency(positive_training_reviews, word_set)
    negative_word_freq_dict = count_word_in_review_frequency(negative_training_reviews, word_set)

    positive_likelihood = calculate_likelihood(positive_word_freq_dict)
    negative_likelihood = calculate_likelihood(negative_word_freq_dict)

    positive_prior = calculate_priors(review_train_data, positive_training_reviews)
    negative_prior = calculate_priors(review_train_data, negative_training_reviews)

    is_positive = classify_text("This movie was really great", positive_likelihood, negative_likelihood, positive_prior,
                                negative_prior)

    if is_positive:
        print("Review is positive")
    else:
        print("Review is negative")

    num_positive = 0
    num_negative = 0
    for index, review in positive_training_reviews.iteritems():
        if classify_text(review, positive_likelihood, negative_likelihood, positive_prior, negative_prior):
            num_positive += 1
        else:
            num_negative += 1

    percent_pos = num_positive / len(positive_test_reviews) * 100

    print("% positive =", percent_pos)

    num_positive = 0
    num_negative = 0
    for index, review in negative_training_reviews.iteritems():
        if classify_text(review, positive_likelihood, negative_likelihood, positive_prior, negative_prior):
            num_positive += 1
        else:
            num_negative += 1

    percent_neg = num_negative / len(negative_test_reviews) * 100

    print("% positive =", percent_neg)

    print("Average % correct =", (percent_pos + percent_neg) / 2)


main()
