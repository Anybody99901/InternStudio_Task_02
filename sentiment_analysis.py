# ============================================================
# SENTIMENT ANALYSIS PROJECT (EDA + MACHINE LEARNING)
# ============================================================

# STEP 1: IMPORT LIBRARIES
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# ============================================================
# STEP 2: LOAD DATASET
# ============================================================

df = pd.read_csv("IMDB Dataset.csv", nrows=10000)
print("Dataset Loaded:", df.shape)

print("\nSample Data:")
print(df.head())

# Convert sentiment to numeric
df['sentiment'] = df['sentiment'].map({'positive':1, 'negative':0})

# ============================================================
# STEP 3: EDA (EXPLORATORY DATA ANALYSIS)
# ============================================================

# Sentiment distribution
plt.figure(figsize=(6,4))
sns.countplot(x='sentiment', data=df)
plt.title("Sentiment Distribution")
plt.xlabel("0 = Negative, 1 = Positive")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("sentiment_distribution.png")
plt.show()

# Review length analysis
df['review_length'] = df['review'].apply(len)

plt.figure(figsize=(8,5))
sns.histplot(df['review_length'], bins=50, kde=True)
plt.title("Review Length Distribution")
plt.xlabel("Length of Review")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("review_length.png")
plt.show()

# ============================================================
# STEP 4: TEXT CLEANING
# ============================================================

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

print("\nCleaning text...")
df['review'] = df['review'].apply(clean_text)

# ============================================================
# STEP 5: MODEL BUILDING
# ============================================================

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2
)

# Convert text to numbers
vectorizer = TfidfVectorizer(max_features=3000)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# ============================================================
# STEP 6: EVALUATION
# ============================================================

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))