import mlflow
import pandas as pd
import mlflow.sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

df = pd.read_csv('IMDB.csv')
df = df.sample(500)
df.to_csv('data.csv', index=False)
df.head()

# data preprocessing

# Define text preprocessing functions
def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(df):
    """Normalize the text data."""
    try:
        df['review'] = df['review'].apply(lower_case)
        df['review'] = df['review'].apply(remove_stop_words)
        df['review'] = df['review'].apply(removing_numbers)
        df['review'] = df['review'].apply(removing_punctuations)
        df['review'] = df['review'].apply(removing_urls)
        df['review'] = df['review'].apply(lemmatization)
        return df
    except Exception as e:
        print(f'Error during text normalization: {e}')
        raise

df = normalize_text(df)
df.head()

df['sentiment'].value_counts()

x = df['sentiment'].isin(['positive','negative'])
df = df[x]

df['sentiment'] = df['sentiment'].map({'positive':1, 'negative':0})
df.head()

df.isnull().sum()

vectorizer = CountVectorizer(max_features=100)
X = vectorizer.fit_transform(df['review'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

import dagshub

mlflow.set_tracking_uri('https://dagshub.com/krunalahir/MLOPS--Capstone-project.mlflow')
dagshub.init(repo_owner='krunalahir', repo_name='MLOPS--Capstone-project', mlflow=True)

# mlflow.set_experiment("Logistic Regression Baseline")
mlflow.set_experiment("Logistic Regression Baseline")

import mlflow
import logging
import os
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logging.info("Starting MLflow run...")

with mlflow.start_run():
    start_time = time.time()

    try:
        logging.info("Logging preprocessing parameters...")
        mlflow.log_param("vectorizer", "Bag of Words")
        mlflow.log_param("num_features", 100)
        mlflow.log_param("test_size", 0.25)

        logging.info("Initializing Logistic Regression model...")
        model = LogisticRegression(max_iter=1000)  # Increase max_iter to prevent non-convergence issues

        logging.info("Fitting the model...")
        model.fit(X_train, y_train)
        logging.info("Model training complete.")

        logging.info("Logging model parameters...")
        mlflow.log_param("model", "Logistic Regression")

        logging.info("Making predictions...")
        y_pred = model.predict(X_test)

        logging.info("Calculating evaluation metrics...")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        logging.info("Logging evaluation metrics...")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        logging.info("Saving and logging the model...")
        mlflow.sklearn.log_model(model, "model")

        # Log execution time
        end_time = time.time()
        logging.info(f"Model training and logging completed in {end_time - start_time:.2f} seconds.")

        # Save and log the notebook
        # notebook_path = "exp1_baseline_model.ipynb"
        # logging.info("Executing Jupyter Notebook. This may take a while...")
        # os.system(f"jupyter nbconvert --to notebook --execute --inplace {notebook_path}")
        # mlflow.log_artifact(notebook_path)

        # logging.info("Notebook execution and logging complete.")

        # Print the results for verification
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"Precision: {precision}")
        logging.info(f"Recall: {recall}")
        logging.info(f"F1 Score: {f1}")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

