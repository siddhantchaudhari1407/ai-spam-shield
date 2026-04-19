# AI-Powered Spam Mail Detection System 🛡️

A comprehensive, professional-grade Spam Detection System built with Python, Scikit-learn, and Streamlit. This project uses Natural Language Processing (NLP) and Machine Learning (ML) to classify emails and messages as Spam or Ham with high precision.

## 🚀 Features
- **Real-time Prediction**: Instantly classify messages with confidence scores.
- **NLP Preprocessing**: Custom pipeline for lowercasing, punctuation removal, stopword filtering, and stemming.
- **Multi-Model Comparison**: Evaluates **Naive Bayes**, **Logistic Regression**, and **SVM**.
- **Interactive Dashboard**: Visual analytics for model accuracy, precision, recall, and F1-score.
- **Bulk Classification**: Upload CSV files for mass prediction.
- **Export & History**: Save prediction logs and export batch results to CSV.
- **Premium UI**: Modern dark-themed interface with glassmorphism and responsive charts.

## 🛠️ Technology Stack
- **Language**: Python 3.x
- **Libraries**:
  - `Scikit-learn`: ML algorithms & evaluation.
  - `Pandas & Numpy`: Data manipulation.
  - `NLTK`: Natural Language Processing.
  - `Streamlit`: Web application framework.
  - `Plotly`: Interactive visualizations.
  - `Joblib`: Model serialization.

## 📁 Project Structure
```text
spam_detection_system/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
├── utils/
│   ├── spam_engine.py      # NLP & Categorization logic
│   └── model_handler.py    # Training & evaluation pipeline
├── data/
│   └── spam.csv            # The SMS Spam Collection dataset
├── models/
│   ├── svm_model.pkl        # Best performing model
│   ├── tfidf_vectorizer.pkl # Fitted vectorizer
│   └── model_metrics.pkl    # Cached results for analytics
└── exports/                # Logs & exported predictions
```

## ⚙️ How to Run
1. **Clone the repository** (or navigate to the directory).
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download Data & Train Models**:
   ```bash
   python data/download_data.py
   python -m utils.model_handler
   ```
4. **Launch the App**:
   ```bash
   streamlit run app.py
   ```

## 📊 Performance
The system currently uses **SVM** as the primary model due to its high accuracy (~97.5%) and robustness on text data. Comparison charts are available in the **Analytics** tab of the application.

---
*Developed as a Second Year Project by Siddhant Chaudhari, Sanchit Shingole, Gunjan Bhangre, Ameya Jadhav*
