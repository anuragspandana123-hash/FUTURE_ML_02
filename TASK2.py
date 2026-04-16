# ===============================
# STAGE 1: Import Libraries
# ===============================
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import pickle

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')

# ===============================
# STAGE 2: Load Dataset
# ===============================
df = pd.read_csv(r"C:\Sales project\TASK2\tickets.csv")
print("Dataset Loaded Successfully")
print(df.head())

# ===============================
# STAGE 3: Text Cleaning (NLP)
# ===============================
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df["clean_ticket"] = df["ticket"].apply(clean_text)

# ===============================
# STAGE 4: Convert Text → Numbers
# ===============================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_ticket"])

y_category = df["category"]
y_priority = df["priority"]

# ===============================
# STAGE 5: Train/Test Split
# ===============================
X_train, X_test, y_train_cat, y_test_cat = train_test_split(
    X, y_category, test_size=0.2, random_state=42)

# ===============================
# STAGE 6: Train Category Model (Naive Bayes)
# ===============================
model_category = MultinomialNB()
model_category.fit(X_train, y_train_cat)

pred_cat = model_category.predict(X_test)
print("\nNaive Bayes Category Accuracy:",
      accuracy_score(y_test_cat, pred_cat))

print("\nDetailed Report:")
print(classification_report(y_test_cat, pred_cat))

# ===============================
# STAGE 7: Train Priority Model
# ===============================
X_train2, X_test2, y_train_pri, y_test_pri = train_test_split(
    X, y_priority, test_size=0.2, random_state=42)

model_priority = MultinomialNB()
model_priority.fit(X_train2, y_train_pri)

pred_pri = model_priority.predict(X_test2)
print("\nPriority Accuracy:",
      accuracy_score(y_test_pri, pred_pri))

# ===============================
# STAGE 8: Try Logistic Regression (Upgrade)
# ===============================
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train_cat)

lr_pred = lr_model.predict(X_test)
print("\nLogistic Regression Accuracy:",
      accuracy_score(y_test_cat, lr_pred))

# ===============================
# STAGE 9: Predict New Ticket
# ===============================
def predict_ticket(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])

    category = model_category.predict(vec)[0]
    priority = model_priority.predict(vec)[0]

    print("\nNew Ticket Prediction")
    print("Category:", category)
    print("Priority:", priority)

predict_ticket("My refund is not processed")

# ===============================
# STAGE 10: Save Models (Production)
# ===============================
pickle.dump(model_category, open("category_model.pkl","wb"))
pickle.dump(model_priority, open("priority_model.pkl","wb"))
pickle.dump(vectorizer, open("vectorizer.pkl","wb"))

print("\nModels saved successfully!")

# ===============================
# STAGE 11: Data Visualization
# ===============================
df["category"].value_counts().plot(kind="bar")
plt.title("Ticket Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()

df["priority"].value_counts().plot(kind="bar")
plt.title("Ticket Priority Distribution")
plt.xlabel("Priority")
plt.ylabel("Count")
plt.show()