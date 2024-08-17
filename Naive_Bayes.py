# -*- coding: utf-8 -*-
"""NaiveBayes_Harish.ipynb
"""
import pandas as pd



df = pd.read_csv('/content/flipkart_product_processed.csv', encoding='latin-1') # Try reading with 'latin-1' encoding

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df.loc[df['Summary'].isna(), 'Summary'] = ''

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df['Summary'], df['Rate'], test_size=0.2, random_state=7)

"""Vectorization (Bag of Words)

CountVectorizer transforms the text data into a matrix of token counts (Bag of Words).
"""

# Vectorize the text data using Bag of Words
vectorizer = CountVectorizer()
x_train_bow = vectorizer.fit_transform(x_train)
x_test_bow = vectorizer.transform(x_test)

"""**Gaussian Naive Bayes**
For numeric data
For large data,
**Multi Nominal Naïve Bayes**
For categorical features like Text data,
**Bernoulli's Naïve Bayes**
For Binary Variables

"""

# Train a Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(x_train_bow, y_train)

# Make predictions
nb_y_pred = nb_model.predict(x_test_bow)

# Evaluate the model
accuracy = accuracy_score(y_test, nb_y_pred)
report = classification_report(y_test, nb_y_pred)
conf_matrix = confusion_matrix(y_test, nb_y_pred)

print(f'Naive Bayes Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')
print(f'Confusion Matrix:\n{conf_matrix}')

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

"""Analyzing misclassified cases helps you understand where your model is making errors

These are the reviews where the **predicted sentiment did not match the actual sentiment.** For example, if a review was **actually positive but the model classified it as negative,** it is a misclassified case.
"""

# Analyze misclassified cases
misclassified = x_test[nb_y_pred != y_test]
print("Sample Misclassified Reviews:")
print(misclassified.head(10))

import seaborn as sns
import matplotlib.pyplot as plt

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

nb_model = MultinomialNB(alpha=0.1, fit_prior=False)
nb_model.fit(x_train_bow, y_train)

# Make predictions
nb_y_pred = nb_model.predict(x_test_bow)

accuracy = accuracy_score(y_test, nb_y_pred)
report = classification_report(y_test, nb_y_pred)
conf_matrix = confusion_matrix(y_test, nb_y_pred)

print(f'Naive Bayes Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')
print(f'Confusion Matrix:\n{conf_matrix}')

nb_model = MultinomialNB(alpha=0.5)
nb_model.fit(x_train_bow, y_train)

accuracy = accuracy_score(y_test, nb_y_pred)
report = classification_report(y_test, nb_y_pred)
conf_matrix = confusion_matrix(y_test, nb_y_pred)

print(f'Naive Bayes Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')
print(f'Confusion Matrix:\n{conf_matrix}')

