# spam-email-detection
# Spam Email Classifier using Python (Scikit-Learn)

## Overview
This project builds a **Spam Email Classifier** using Python and **Scikit-Learn**. It preprocesses text data, trains a **Naïve Bayes model**, and allows real-time testing of emails to classify them as **Spam or Ham**.

---

## 📌 Steps in the Code

1. **Load & Preprocess Dataset**
2. **Balance Data if Necessary**
3. **Split Data into Training & Testing Sets**
4. **Build a Naïve Bayes Classifier using a Pipeline**
5. **Train and Evaluate the Model**
6. **Perform Real-Time Email Classification**

---

## 🚀 Full Code Implementation
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
try:
    df = pd.read_csv("spam_ham_dataset.csv")
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: Dataset file not found. Please check the file path.")
    exit()

# Ensure correct columns exist
if 'label' not in df.columns or 'text' not in df.columns:
    print("Error: Expected columns 'label' and 'text' not found in dataset.")
    exit()

# Preprocess dataset
df = df[['label', 'text']]
df.columns = ['labels', 'message']
df['labels'] = df['labels'].map({'ham': 0, 'spam': 1})  # Convert spam=1, ham=0

# Check dataset balance
print("\nDataset Distribution:\n", df['labels'].value_counts())

# Balance dataset if needed (Duplicate spam samples if spam count is very low)
spam_count = df[df['labels'] == 1].shape[0]
ham_count = df[df['labels'] == 0].shape[0]

if spam_count < ham_count:
    extra_spam = df[df['labels'] == 1].sample(ham_count - spam_count, replace=True, random_state=42)
    df = pd.concat([df, extra_spam])

print("\nBalanced Dataset Distribution:\n", df['labels'].value_counts())

# Step 3: Splitting Data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['labels'], test_size=0.2, random_state=42)
print(f"\nTraining set size: {len(X_train)} messages")
print(f"Testing set size: {len(X_test)} messages")

# Step 4: Build the Model Pipeline
model = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.9, min_df=2)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

# Train the Model
model.fit(X_train, y_train)
print("\nModel training completed!")

# Evaluate Model Performance
y_pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 5: Real-Time Testing
print("\n🔹 Real-Time Email Classification 🔹")
while True:
    user_input = input("\nEnter an email message (or type 'exit' to stop): ")
    if user_input.lower() == 'exit':
        print("Exiting real-time testing.")
        break
    
    # Get probability scores
    probabilities = model.predict_proba([user_input])[0]
    
    # Set a stricter threshold for spam detection
    spam_threshold = 0.3  # Adjust if needed
    prediction = 1 if probabilities[1] > spam_threshold else 0

    print("Prediction:", "🚨 Spam" if prediction == 1 else "✅ Ham")
```

---

## 🔹 Example Test Cases
```plaintext
Enter an email message (or type 'exit' to stop): 🎉 You won a $1000 Amazon gift card! Click here to claim now.
Prediction: 🚨 Spam

Enter an email message (or type 'exit' to stop): Hey, let's meet for coffee at 5 PM.
Prediction: ✅ Ham

Enter an email message (or type 'exit' to stop): URGENT! Your PayPal account is compromised.
Prediction: 🚨 Spam
```

---

## 🎯 Summary
- This script **trains a spam classifier** using **Naïve Bayes**.
- It **preprocesses the dataset**, balances it if necessary, and **evaluates model accuracy**.
- The model can be **tested in real time** by entering email messages.

💡 **Adjust `spam_threshold` to fine-tune spam detection sensitivity!** 🚀
