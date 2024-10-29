# Step 1: Import Necessary Libraries

# Import basic libraries
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# For text preprocessing
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# Step 2: Load the Dataset

# Load the dataset
file_path = 'data/majordatasetwithsubcalass.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
df.head()


# Step 3: Check Data Overview
# Let's understand the dataset structure and check for any missing values.

# Check basic information about the dataset
df.info()

# Check for missing values
df.isnull().sum()

# Drop rows with missing values (if any)
df.dropna(inplace=True)


# Step 4: Define a Function to Clean Marathi Text
# We define a function to remove any non-Marathi characters from the text data. This step is critical to ensure that only Marathi language characters are present.

# Function to clean the text by removing non-Marathi characters
def clean_marathi_text(text):
    # Use a regular expression to remove anything that is not a Marathi character
    marathi_only = re.sub(r'[^\u0900-\u097F\s]', '', text)  # Unicode range for Marathi characters
    # Remove extra spaces
    marathi_only = re.sub(r'\s+', ' ', marathi_only).strip()
    return marathi_only

# Apply the cleaning function to the 'Text' column
df['cleaned_text'] = df['Text'].apply(clean_marathi_text)

# Display the cleaned text
df[['Text', 'cleaned_text']].head()


# Step 5: Remove Stopwords (Optional)
# If you have a list of Marathi stopwords (common words that don't contribute much to the meaning), you can remove them to further clean the text.

# Load Marathi stopwords from NLTK or define your own list of stopwords
marathi_stopwords = set(stopwords.words('C:\HSD major project\data\marathi_stopwords.txt'))

# Function to remove stopwords from the textp
def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in marathi_stopwords])

# Apply stopword removal
df['cleaned_text'] = df['cleaned_text'].apply(remove_stopwords)

# Display the cleaned text after removing stopwords
df[['cleaned_text']].head()

#
# Step 5: Split Data into Features and Labels
# We split the dataset into features (cleaned text) and two labels: Label (binary classification) and Subclass (multi-class classification).

# Features (X) and Labels (y_label, y_subclass)
X = df['cleaned_text']  # Cleaned text as features
y_label = df['Label']   # Binary label (hate speech or not)
y_subclass = df['Subclass']  # Multi-class label (type of hate speech)

# Split the data into training and testing sets
X_train, X_test, y_label_train, y_label_test, y_subclass_train, y_subclass_test = train_test_split(
    X, y_label, y_subclass, test_size=0.2, random_state=42
)

# Print data sizes
print(f'Training set size: {X_train.shape[0]}')
print(f'Testing set size: {X_test.shape[0]}')


# Step 6: Vectorize Text Using TF-IDF
# We will vectorize the text using TF-IDF for numerical representation.

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Check shape
print(f'Train TF-IDF shape: {X_train_tfidf.shape}')
print(f'Test TF-IDF shape: {X_test_tfidf.shape}')


# Step 7: Train Logistic Regression for Label (Binary Classification)
# Train a Logistic Regression model to classify the Label (hate speech or not).

# Train Logistic Regression for binary classification
lr_label = LogisticRegression(max_iter=1000)
lr_label.fit(X_train_tfidf, y_label_train)

# Predict on test set
y_label_pred = lr_label.predict(X_test_tfidf)

# Evaluate the binary classifier
print("Binary Classification (Label) Report:\n", classification_report(y_label_test, y_label_pred))
print("Accuracy for Label:", accuracy_score(y_label_test, y_label_pred))


# Step 8: Train Logistic Regression for Subclass (Multi-class Classification)
# Now, we train another Logistic Regression model to classify the

# Train Logistic Regression for subclass (multi-class classification)
lr_subclass = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
lr_subclass.fit(X_train_tfidf, y_subclass_train)

# Predict on test set
y_subclass_pred = lr_subclass.predict(X_test_tfidf)

# Evaluate the multi-class classifier
print("Multi-class Classification (Subclass) Report:\n", classification_report(y_subclass_test, y_subclass_pred))
print("Accuracy for Subclass:", accuracy_score(y_subclass_test, y_subclass_pred))




# Step 9: Joint Prediction and Output
# We can now jointly predict both Label and Subclass and output them together.

# Make predictions for both binary Label and Subclass
y_label_pred = lr_label.predict(X_test_tfidf)
y_subclass_pred = lr_subclass.predict(X_test_tfidf)

# Combine the results into a DataFrame for easier visualization
results = pd.DataFrame({
    'Text': X_test,
    'Predicted_Label': y_label_pred,
    'Actual_Label': y_label_test,
    'Predicted_Subclass': y_subclass_pred,
    'Actualpy_Subclass': y_subclass_test
})

# Display the results
results.head()


# Step 10: Saving Models and Vectorizer
# Save the models and the vectorizer for deployment.

import pickle

# Save the Logistic Regression models and TF-IDF vectorizer
with open('models/lr_label_model.pkl', 'wb') as label_file:
    pickle.dump(lr_label, label_file)

with open('models/lr_subclass_model.pkl', 'wb') as subclass_file:
    pickle.dump(lr_subclass, subclass_file)

with open('models/tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)
