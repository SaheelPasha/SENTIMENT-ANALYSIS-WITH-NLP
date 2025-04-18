#Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#Sample Dataset (Expanded for better balance)

data = {
    'review': [
        'I love this product, it is amazing!',
        'Terrible experience, will not buy again.',
        'Very satisfied with the purchase.',
        'Worst product ever!',
        'Absolutely fantastic!',
        'Not good, not recommended.',
        'I am happy with this item.',
        'It broke after one use, very bad quality.',
        'Excellent quality and performance.',
        'Waste of money.',
        'Very bad quality and terrible service.',
        'Worst purchase ever made.',
        'Completely useless and broke in one day.',
        'Disappointing and poorly built.',
        'Not worth a single penny. Hated it.',
        'Amazing experience, will buy again!',
        'Loved everything about this product!',
        'Top-notch quality and works great.'
    ],
    'sentiment': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0,0, 0, 0, 0, 0, 1, 1, 1]  # 1 = Positive, 0 = Negative
}

#Converting data into a DataFrame
df = pd.DataFrame(data)

#Splitting the data into train and test sets
X = df['review']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Text Vectorization using TF-IDF with n-grams and full word coverage
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 3),   # unigrams, bigrams, trigrams
    min_df=1              # include rare terms
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#Logistic Regression Model with class balance and more iterations
model = LogisticRegression(class_weight='balanced', max_iter=200)
model.fit(X_train_tfidf, y_train)

#Making Predictions
y_pred = model.predict(X_test_tfidf)

#Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

#Testing on Custom Reviews
custom_reviews = ["I hate this thing", "Works perfectly!", "Very bad quality", "Loved it so much"]
custom_vector = vectorizer.transform(custom_reviews)
predictions = model.predict(custom_vector)

#Showing results
print("\nCustom Review Predictions:")
for review, pred in zip(custom_reviews, predictions):
    sentiment = "Positive" if pred == 1 else "Negative"
    print(f"Review: {review} -> Sentiment: {sentiment}")
