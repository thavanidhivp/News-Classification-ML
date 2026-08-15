# ============================================
# NEWS CLASSIFICATION USING MACHINE LEARNING
# ============================================

# Import libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================
# 1. LOAD DATASET
# ============================================

df = pd.read_csv("news_dataset.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Display first 5 records
print("\nFirst 5 records:")
print(df.head())


# ============================================
# 2. CHECK DATASET
# ============================================

print("\nColumn names:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())


# ============================================
# 3. HANDLE MISSING VALUES
# ============================================

df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")

# Remove rows where label is missing
df = df.dropna(subset=["label"])


# ============================================
# 4. COMBINE TITLE AND TEXT
# ============================================

df["content"] = df["title"] + " " + df["text"]


# ============================================
# 5. DEFINE INPUT AND TARGET
# ============================================

X = df["content"]
y = df["label"]

print("\nClass distribution:")
print(y.value_counts())


# ============================================
# 6. SPLIT DATASET
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================
# 7. TF-IDF FEATURE EXTRACTION
# ============================================

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF feature extraction completed.")
print("Training feature shape:", X_train_tfidf.shape)
print("Testing feature shape:", X_test_tfidf.shape)


# ============================================
# 8. TRAIN LOGISTIC REGRESSION MODEL
# ============================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

print("\nModel training completed successfully!")


# ============================================
# 9. MAKE PREDICTIONS
# ============================================

y_pred = model.predict(X_test_tfidf)


# ============================================
# 10. MODEL EVALUATION
# ============================================

accuracy = accuracy_score(y_test, y_pred)

print("\n============================================")
print("           MODEL EVALUATION")
print("============================================")

print("\nAccuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================
# 11. TEST WITH NEW NEWS ARTICLE
# ============================================

sample_news = [
    "The government announced new economic policies to improve the country's growth."
]

sample_tfidf = tfidf.transform(sample_news)

prediction = model.predict(sample_tfidf)

print("\n============================================")
print("           NEW NEWS PREDICTION")
print("============================================")

print("News:")
print(sample_news[0])

print("\nPredicted Label:")
print(prediction[0])


# ============================================
# END OF PROGRAM
# ============================================
