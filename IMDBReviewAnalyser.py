# Name: Graeme Hosford
# Student ID: R00147327

import math
from collections import Counter

import pandas as pd
from sklearn import model_selection, metrics


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

    for index, review in review_list.iteritems():
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


def evaluate_results(positive_training_dataset: pd.Series, negative_training_dataset: pd.Series,
                     positive_label_dataset: pd.Series, negative_label_dataset: pd.Series,
                     positive_test_dataset: pd.Series, negative_test_dataset: pd.Series,
                     full_training_dataset: pd.Series, min_word_occurance: int):
    pos_train_length = len(positive_training_dataset)
    neg_train_length = len(negative_training_dataset)
    pos_test_length = len(positive_test_dataset)
    neg_test_length = len(negative_test_dataset)

    print("Number of positive training reviews:", pos_train_length)
    print("Number of negative training reviews:", neg_train_length)
    print("Number of positive test reviews:", pos_test_length)
    print("Number of negative test reviews:", neg_test_length)

    full_label_dataset = positive_label_dataset.append(negative_label_dataset)

    kfolds = model_selection.KFold(n_splits=4)
    accuracies_for_lengths = []

    for word_length in range(1, 10):
        print("Checking word length:", word_length)
        average_accuracies = []
        for train_index, test_index in kfolds.split(full_training_dataset):
            train_subset = full_training_dataset.iloc[train_index]
            test_subset = full_training_dataset.iloc[test_index]

            cleaned_train_subset = clean_data(train_subset)

            word_set = count_training_words(cleaned_train_subset, word_length, min_word_occurance)

            positive_word_freq_dict = count_word_in_review_frequency(positive_training_dataset, word_set)
            negative_word_freq_dict = count_word_in_review_frequency(negative_training_dataset, word_set)

            positive_prior = calculate_priors(cleaned_train_subset, positive_training_dataset)
            negative_prior = calculate_priors(cleaned_train_subset, negative_training_dataset)

            positive_likelihood = calculate_likelihood(positive_word_freq_dict)
            negative_likelihood = calculate_likelihood(negative_word_freq_dict)

            results = []
            for index, review in positive_test_dataset.iteritems():
                result = classify_text(review, positive_likelihood, negative_likelihood,
                                       positive_prior, negative_prior)
                if result:
                    results.append("positive")
                else:
                    results.append("negative")

            for index, review in negative_test_dataset.iteritems():
                result = classify_text(review, positive_likelihood, negative_likelihood,
                                       positive_prior, negative_prior)
                if result:
                    results.append("positive")
                else:
                    results.append("negative")

            accuracy = metrics.accuracy_score(full_label_dataset, results)

            average_accuracies.append(accuracy)
            accuracy_percent = accuracy * 100

            print("Accuracy:", accuracy_percent, "%")

        average_accuracy = sum(average_accuracies) / 4
        accuracies_for_lengths.append(average_accuracy)
        print("Average Accuracy:", str(average_accuracy * 100), "%")

    best_average = None
    best_word_length = None

    for index, average in enumerate(accuracies_for_lengths, 1):
        if best_average is None:
            best_average = average
            best_word_length = index
        elif best_average < average:
            best_average = average
            best_word_length = index

    clean_positive_review_data = clean_data(positive_training_dataset)
    clean_negative_review_data = clean_data(negative_training_dataset)

    word_set_for_best = count_training_words(full_training_dataset, best_word_length, min_word_occurance)

    positive_word_dict = count_word_in_review_frequency(clean_positive_review_data, word_set_for_best)
    negative_word_dict = count_word_in_review_frequency(clean_negative_review_data, word_set_for_best)

    positive_prior = calculate_priors(full_training_dataset, clean_positive_review_data)
    negative_prior = calculate_priors(full_training_dataset, clean_negative_review_data)

    positive_likelihood = calculate_likelihood(positive_word_dict)
    negative_likelihood = calculate_likelihood(negative_word_dict)

    results = []
    for index, review in positive_test_dataset.iteritems():
        result = classify_text(review, positive_likelihood, negative_likelihood, positive_prior, negative_prior)
        if result:
            results.append("positive")
        else:
            results.append("negative")

    for index, review in negative_test_dataset.iteritems():
        result = classify_text(review, positive_likelihood, negative_likelihood, positive_prior, negative_prior)
        if result:
            results.append("positive")
        else:
            results.append("negative")

    length_test_data = len(positive_test_dataset) + len(negative_test_dataset)
    accuracy = metrics.accuracy_score(full_label_dataset, results)
    confusion_matrix = metrics.confusion_matrix(full_label_dataset, results)
    percentage_true_pos = (confusion_matrix[0, 0] / length_test_data) * 100
    percentage_false_pos = (confusion_matrix[0, 1] / length_test_data) * 100
    percentage_true_neg = (confusion_matrix[1, 1] / length_test_data) * 100
    percentage_false_neg = (confusion_matrix[1, 0] / length_test_data) * 100

    print("Test data accuracy:", str(accuracy * 100) + "%")
    print("Confusion matrix:", str(confusion_matrix))
    print("Percentage True Positives", str(percentage_true_pos) + "%")
    print("Percentage False Positives", str(percentage_false_pos) + "%")
    print("Percentage True Negatives", str(percentage_true_neg) + "%")
    print("Percentage False Negatives", str(percentage_false_neg) + "%")


def main():
    review_train_data, label_train_data, review_test_data, label_test_data = split_data()

    review_train_data = clean_data(review_train_data)

    positive_training_reviews = review_train_data[label_train_data == "positive"]
    negative_training_reviews = review_train_data[label_train_data == "negative"]
    positive_test_reviews = review_test_data[label_test_data == "positive"]
    negative_test_reviews = review_test_data[label_test_data == "negative"]

    positive_label_data = label_test_data[label_test_data == "positive"]
    negative_label_data = label_test_data[label_test_data == "negative"]

    evaluate_results(positive_training_reviews, negative_training_reviews, positive_label_data,
                     negative_label_data, positive_test_reviews,
                     negative_test_reviews, review_train_data, 5)


main()
