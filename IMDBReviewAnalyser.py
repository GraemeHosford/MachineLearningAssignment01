# Name: Graeme Hosford
# Student ID: R00147327

import numpy as np
import pandas as pd
import re

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

for row in review_training_data:
    text = re.sub("[^0-9a-zA-Z ]+", "", str(row)).lower()
    print("\n\n", text)
