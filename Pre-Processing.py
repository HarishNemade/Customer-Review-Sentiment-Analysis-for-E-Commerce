# -*- coding: utf-8 -*-
"""ML_Project.ipynb

#Data Collection and Pre-Processing
"""

import pandas as pd

"""latin-1 is likely being used because the data might contain Western European characters, and it avoids errors that could occur with other encodings like UTF-8."""

df = pd.read_csv("/content/flipkart_product.csv", encoding='latin-1')  # Or try 'iso-8859-1', 'utf-16', etc.

df.head()

"""Check Null Values

"""

df.isnull().sum()

"""Price only have 1 null value so we will remove it
Rate also have 1 null value so remove it
for review and summary we will fill null values as no review and no summary accordingly.
"""

df["Review"] = df["Review"].fillna("No Review")
df["Summary"] = df["Summary"].fillna("No Summary")

"""drop null value for price and rate contant 1 value which is null"""

df.isnull().sum()

df = df.dropna()

df.isnull().sum()

df.info()

df.shape

"""Check Duplication and drop it"""

df.duplicated().sum()

df.drop_duplicates(inplace=True)

df.shape

"""NLP Processing"""

import nltk # Natural Language Toolkit (NLTK)  Natural Language Toolkit (NLTK)
nltk.download('stopwords') #filtered out before processing text because they do not contribute much to the meaning ("is","the","and")
from nltk.corpus import stopwordsfrom nltk.tokenize import word_tokenize #splits a string into individual words (tokens).
nltk.download('punkt')

df.head()

"""Defining the remove_stop_words Function:"""

def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))    #This line loads the list of English stopwords into a set (for faster lookup).
    word_tokens = word_tokenize(text)  #This tokenizes the input text into individual words
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]    #This line filters out any word that is a stopword
    return ' '.join(filtered_text)  #Finally, the filtered words are joined back into a single string and returned.

"""*df['Summary']*.astype(str):   Ensures that the Summary column is treated as strings (in case some entries are not strings).
*.apply(remove_stop_words):* Applies the remove_stop_words function to each element in the Summary column, effectively removing stopwords from the text.
"""

df['Summary'] = df['Summary'].astype(str).apply(remove_stop_words)

df.head()

"""Wordcloud"""

!pip install wordcloud matplotlib

from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = " ".join(df['Summary'].tolist())

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(30, 10))
plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis("off")
plt.show()

"""1. Sentiment Analysis Function"""

!pip install -U textblob

"""TextBlob is a Python library for easy text processing, providing tools for sentiment analysis, text classification, part-of-speech tagging, translation, and more with simple and intuitive APIs."""

from textblob import TextBlob

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df['sentiment'] = df['Summary'].apply(get_sentiment)

Review_sentiment = df.groupby('Review')['sentiment'].mean().reset_index()
print(Review_sentiment)

Price_sentiment = df.groupby('Price')['sentiment'].mean().reset_index()
print(Price_sentiment)

Product_sentiment = df.groupby('ProductName')['sentiment'].mean().reset_index()
print(Product_sentiment)

import matplotlib.pyplot as plt
# Create masks for positive and negative words
positive_mask = df['sentiment'] > 0
negative_mask = df['sentiment'] < 0

# Generate wordclouds
positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(df[positive_mask]['Summary']))
negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(df[negative_mask]['Summary']))

# Display wordclouds
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.title("Positive Words")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.title("Negative Words")
plt.axis("off")

plt.show()

"""pie chart to visualize the distribution of positive and negative sentiments"""

import matplotlib.pyplot as plt

# Count positive and negative sentiments
positive_count = df[df['sentiment'] > 0]['sentiment'].count()
negative_count = df[df['sentiment'] < 0]['sentiment'].count()

# Create pie chart
labels = ['Positive', 'Negative']
sizes = [positive_count, negative_count]
explode = (0.1, 0)  # Explode the first slice (Positive)



#%1.1f%% means it will display the percentage with one decimal place.

plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Sentiment Distribution')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 5))
ax = sns.barplot(x='Review', y='sentiment', data=Review_sentiment,
                 palette=sns.color_palette("coolwarm", n_colors=len(Review_sentiment))) # Color the bars based on sentiment
plt.title('Average Sentiment by Review')
plt.xlabel('Review')
plt.ylabel('Average Sentiment')

# Add color bar
norm = plt.Normalize(df['sentiment'].min(), df['sentiment'].max())
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
sm.set_array([])  # This is necessary for the colorbar to work
plt.colorbar(sm, orientation='vertical', label='Sentiment Range')

plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Sort by sentiment and select the top 10 reviews
top_20_reviews = Review_sentiment.sort_values(by='sentiment', ascending=False).head(20)

# Plotting with color palette
plt.figure(figsize=(20, 5))
ax = sns.barplot(x='Review', y='sentiment', data=top_20_reviews,
                 palette=sns.color_palette("coolwarm", n_colors=len(top_20_reviews))) # Color the bars based on sentiment
plt.title('Top 20 Reviews by Average Sentiment')
plt.xlabel('Review')
plt.ylabel('Average Sentiment')

# Add color bar
# Add color bar
norm = plt.Normalize(1, -1)  # Set color bar range from -1 to 1
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
sm.set_array([])  # This is necessary for the colorbar to work
plt.colorbar(sm, orientation='vertical', label='Sentiment Range')

plt.xticks(rotation=90)  # Rotate x-axis labels for readability

plt.show()

#Price sentiment

plt.figure(figsize=(20, 5))
ax = sns.barplot(x='Price', y='sentiment', data=Price_sentiment,
                 palette=sns.color_palette("coolwarm", n_colors=len(Price_sentiment))) # Color the bars based on sentiment
plt.title('Average Sentiment by Price')
plt.xlabel('Price')
plt.ylabel('Average Sentiment')

# Add color bar
norm = plt.Normalize(df['sentiment'].min(), df['sentiment'].max())
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
sm.set_array([])
plt.colorbar(sm, orientation='vertical', label='Sentiment Range')

plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Sort by sentiment and select the top 20 entries
top_20_prices = Price_sentiment.sort_values(by='sentiment', ascending=False).head(20)

# Plotting with color palette
plt.figure(figsize=(20, 5))
ax = sns.barplot(x='Price', y='sentiment', data=top_20_prices,
                 palette=sns.color_palette("coolwarm", n_colors=len(top_20_prices))) # Color the bars based on sentiment
plt.title('Top 20 Prices by Average Sentiment')
plt.xlabel('Price')
plt.ylabel('Average Sentiment')

# Add color bar
norm = plt.Normalize(-1, 1)  # Set color bar range from -1 to 1
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
sm.set_array([])  # This is necessary for the colorbar to work
plt.colorbar(sm, orientation='vertical', label='Sentiment Range')

plt.xticks(rotation=90)  # Rotate x-axis labels for readability

plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Sort the Product_sentiment DataFrame by sentiment and select the top 20
top_20_products = Product_sentiment.sort_values(by='sentiment', ascending=False).head(20)

# Plotting the top 20 products
plt.figure(figsize=(20, 5))
ax = sns.barplot(x='ProductName', y='sentiment', data=top_20_products,
                 palette=sns.color_palette("coolwarm", n_colors=len(top_20_products)))
plt.title('Top 20 Products by Average Sentiment')
plt.xlabel('Product Name')
plt.ylabel('Average Sentiment')
plt.xticks(rotation=90)  # Rotate x-axis labels for readability

# Add color bar
norm = plt.Normalize(-1, 1)  # Set color bar range from -1 to 1
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
sm.set_array([])  # This is necessary for the colorbar to work
plt.colorbar(sm, orientation='vertical', label='Sentiment Range')

plt.xticks(rotation=90)  # Rotate x-axis labels for readability

plt.show()

"""Create the Rate Column on Categorical if rate is greater 3 then 1 else 0

errors='coerce' Any values that cannot be converted (e.g., strings or other non-numeric data) are turned into NaN (Not a Number).
"""

# Handle non-numeric values in the 'Rate' column
df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce')

# Convert valid ratings to binary (1 if > 3, else 0)
df['Rate'] = df['Rate'].apply(lambda x: 1 if x > 3 else 0)

df.head()

df.to_csv('flipkart_product_processed.csv', index=False)
from google.colab import files
files.download('flipkart_product_processed.csv')

"""##Word2Vec"""

df.isna().sum()

df.loc[df['Summary'].isna(), 'Summary'] = ''  # Replace NaN values in the Summary column with an empty string text using loc.

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

"""Tokenize the text data"""

!pip install nltk
import nltk
nltk.download('punkt')

df['tokenized_summary'] = df['Summary'].apply(word_tokenize)

print(df['tokenized_summary'])

# Train Word2Vec model on the tokenized text
model = Word2Vec(sentences=df['tokenized_summary'], vector_size=100, window=5, min_count=1, workers=4)

# Train Word2Vec model on the tokenized text
import numpy as np
model = Word2Vec(sentences=df['tokenized_summary'], vector_size=100, window=5, min_count=1, workers=4)

# Generate feature vectors for each review by averaging the Word2Vec vectors of the words in the review.
def get_vector(tokens, model):
    # Get vectors for each token
    vectors=[model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        # Calculate the mean of the vectors
        return np.mean(vectors, axis=0)
    else:
        # Return a zero vector if there are no vectors for the words
        return np.zeros(model.vector_size)

# Generate feature vectors for all reviews
# Use the 'model' variable here instead of 'w2v'
df['feature_vector']=df['tokenized_summary'].apply(lambda x: get_vector(x, model))

# Convert the list of arrays into a 2D array
x=np.vstack(df['feature_vector'])

# Use the 'Rate' column as the target variable
y=df['Rate']

print("Feature vectors (x):")
print(x)
print("Target variable (y):")
print(y)

print(x.shape)
print(y.shape)

# Split the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=7)

import xgboost as xgb

# Initialize the XGBoost model
model=xgb.XGBClassifier()

# Train the model
model.fit(x_train, y_train)

"""Support:

Class 0 (7884): There are 7884 reviews that are actually class 0.
Class 1 (25119): There are 25119 reviews that are actually class 1.
Macro Average:

Precision (87%): Average precision across both classes, treating each class equally.
Recall (80%): Average recall across both classes, treating each class equally.
F1-Score (83%): Average F1-score across both classes, treating each class equally.
Weighted Average:

Precision (88%): Average precision considering the number of reviews in each class.
Recall (89%): Average recall considering the number of reviews in each class.
F1-Score (88%): Average F1-score considering the number of reviews in each class.
Summary of Output:
Overall Performance: The model is quite good with an accuracy of 88.54%. It’s very good at identifying class 1 reviews but not as good with class 0 reviews.
Class-Specific Performance: It’s better at predicting class 1 (with high precision and recall) compared to class 0 (where recall is lower).
Balance: The macro average shows the model’s overall performance across all classes, while the weighted average takes into account how many reviews there are for each class.
The model is performing well overall, especially with class 1, but there is room for improvement in identifying class 0 reviews.
"""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predict the labels for the test set
y_pred=model.predict(x_test)

# Calculate accuracy
accuracy=accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Print classification report
print(classification_report(y_test, y_pred))

print(y_pred)

import matplotlib.pyplot as plt
import seaborn as sns

# Predict on the test data
y_pred = model.predict(x_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
print(report)

# Step 4: Error Analysis
# Identify misclassified examples
misclassified = x_test[(y_test.values.flatten() != y_pred)]

print(f'Misclassified examples: {len(misclassified)}')
# Analyze a few misclassified examples
print(misclassified.argsort()[-10])

# Display the corresponding true labels and predictions
print(f'True labels of misclassified examples: {y_test[(y_test.values.flatten() != y_pred)].values.flatten()}')
print(f'Predictions of misclassified examples: {y_pred[(y_test.values.flatten() != y_pred)]}')

# Check class distribution in training data
train_class_distribution = y_train.value_counts(normalize=True)
print("Training Class Distribution:\n", train_class_distribution)

# Check class distribution in test data
test_class_distribution = y_test.value_counts(normalize=True)
print("Test Class Distribution:\n", test_class_distribution)

from imblearn.over_sampling import RandomOverSampler

# Perform random oversampling to balance the classes
oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
x_train_resampled, y_train_resampled = oversampler.fit_resample(x_train, y_train)

from imblearn.under_sampling import RandomUnderSampler

# Perform random undersampling to balance the classes
undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
x_train_resampled, y_train_resampled = undersampler.fit_resample(x_train, y_train)

# Define the XGBoost model
model = xgb.XGBClassifier(random_state=42)

# Train the model with resampled data
model.fit(x_train_resampled, y_train_resampled)

# Predict on the test data
y_pred = model.predict(x_test)

# Classification Report
report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
print("Classification Report:\n", report)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report Breakdown
# Accuracy (84%):

# Meaning: Overall, the model correctly predicted the sentiment of 84% of the reviews.
# Precision:

# Negative (64%): When the model predicted a review as "Negative," it was correct 64% of the time.
# Positive (93%): When the model predicted a review as "Positive," it was correct 93% of the time.
# Recall:

# Negative (79%): Out of all actual "Negative" reviews, the model correctly identified 79% of them.
# Positive (86%): Out of all actual "Positive" reviews, the model correctly identified 86% of them.
# F1-Score:

# Negative (70%): This is a combined measure of precision and recall for "Negative" reviews. It shows how well the model did in identifying "Negative" reviews, balancing precision and recall.
# Positive (89%): This is a combined measure of precision and recall for "Positive" reviews. It shows how well the model did in identifying "Positive" reviews, balancing precision and recall.
# Support:

# Negative (7884): There are 7884 reviews that are actually "Negative."
# Positive (25119): There are 25119 reviews that are actually "Positive."
# Macro Average:

# Precision (78%): The average precision across both "Negative" and "Positive" classes, treating each class equally.
# Recall (82%): The average recall across both classes, treating each class equally.
# F1-Score (80%): The average F1-score across both classes, treating each class equally.
# Weighted Average:

# Precision (86%): The average precision considering the number of reviews in each class. It reflects the performance adjusted for class imbalance.
# Recall (84%): The average recall considering the number of reviews in each class.
# F1-Score (85%): The average F1-score considering the number of reviews in each class.
# Summary
# Overall Performance: The model is good overall with an accuracy of 84%. It is particularly strong at identifying "Positive" reviews (93% precision and 86% recall).
# Class-Specific Performance: The model performs better with "Positive" reviews compared to "Negative" reviews.
# Balance: The macro average shows the model's performance averaged across classes, while the weighted average adjusts for the number of instances in each class, providing a more balanced view of performance.

