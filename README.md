# SENTIMENT-ANALYSIS-WITH-NLP

**COMPANY**: CODTECH IT SOLUTIONS

**NAME**: Saheel Pasha

**INTERN ID**: CT04WT62

**DOMAIN**:  Machine Learning

**DURATION**: 4 WEEKS

**MENTOR**: NEELA SANTOSH

# DESCRIPTION
This task involves performing sentiment analysis on a dataset of customer reviews using Natural Language Processing (NLP) techniques, specifically leveraging TF-IDF vectorization and a Logistic Regression model for classification.

# Overview of the Process:

# Dataset Preparation:
A sample dataset is manually defined with customer reviews labeled as either positive (1) or negative (0).
This dataset is structured using a Pandas DataFrame to facilitate preprocessing and model training.

# Data Splitting:
The reviews and their sentiments are split into training and testing sets using train_test_split from Scikit-learn.
80% of the data is used for training and 20% for testing.

# Text Vectorization with TF-IDF:
Text data is converted into numerical format using TF-IDF Vectorization (TfidfVectorizer), which helps in emphasizing important words in each review.
The vectorizer is configured to consider unigrams, bigrams, and trigrams (ngram_range=(1, 3)) for capturing richer text features.
Common stop words are removed to reduce noise.

# Model Building with Logistic Regression:
A Logistic Regression classifier is trained on the vectorized training data.
The model uses class_weight='balanced' to handle any class imbalance in the dataset.
max_iter=200 ensures the solver has enough iterations to converge during training.

# Model Evaluation:
After training, the model is tested on the test set.
Evaluation metrics such as accuracy score, confusion matrix, and a classification report (precision, recall, f1-score) are printed for performance analysis.

# Testing on Custom Inputs:
The model is further tested on new, unseen reviews to evaluate how well it generalizes.
Sentiments are predicted and printed as either "Positive" or "Negative" based on model inference.

# Outcome:
This task demonstrates the end-to-end application of an NLP pipeline, covering preprocessing, vectorization, classification, and evaluation. It builds practical skills in applying machine learning for sentiment analysis, an essential technique in understanding customer feedback.

# output


