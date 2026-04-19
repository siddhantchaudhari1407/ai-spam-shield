import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utils.spam_engine import clean_text

MODELS_DIR = "models"
DATA_PATH = "data/spam.csv"

def train_models():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    # Load dataset
    if not os.path.exists(DATA_PATH):
        print("Dataset not found. Please run download_data.py first.")
        return None
    
    df = pd.read_csv(DATA_PATH)
    
    # Preprocess
    print("Preprocessing data...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Label encoding
    df['target'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['target'], test_size=0.2, random_state=42
    )
    
    # Vectorization
    print("Vectorizing text (Bigrams enabled)...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Define models with class balancing
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(solver='liblinear', class_weight='balanced'),
        "SVM": SVC(probability=True, class_weight='balanced', kernel='linear')
    }
    
    results = {}
    best_accuracy = 0
    best_model_name = ""
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm
        }
        
        # Save each model
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name.lower().replace(' ', '_')}_model.pkl"))
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            
    # Save vectorizer
    joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    
    # Save metadata/results
    joblib.dump(results, os.path.join(MODELS_DIR, "model_metrics.pkl"))
    
    print(f"Best model: {best_model_name} with {best_accuracy:.4f} accuracy")
    return results

if __name__ == "__main__":
    train_models()
