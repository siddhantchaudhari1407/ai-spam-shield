import streamlit as st
import pandas as pd
import joblib
import os
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from utils.spam_engine import clean_text, categorize_spam
from utils.model_handler import train_models

# Page Config
st.set_page_config(
    page_title="AI Spam Detector Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Premium Dark Pro)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    .main {
        background: radial-gradient(circle at top right, #1a1a2e, #16213e, #0f0c29) !important;
        color: #ffffff;
    }
    .stApp {
        background: #0f0c29;
    }
    [data-testid="stSidebar"] {
        background-color: #1a1a2e;
        border-right: 1px solid rgba(233, 69, 96, 0.2);
    }
    /* Buttons with Neon Glow */
    .stButton>button {
        background: linear-gradient(90deg, #e94560, #ff2e63);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.3);
        width: 100%;
    }
    .stButton>button:hover {
        transform: scale(1.03) translateY(-2px);
        box-shadow: 0 0 25px rgba(233, 69, 96, 0.6);
        color: white;
    }
    /* Advanced Glassmorphism Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        padding: 25px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        transition: all 0.3s ease;
        text-align: center;
    }
    .metric-card:hover {
        transform: translateY(-8px);
        border: 1px solid rgba(233, 69, 96, 0.4);
        background: rgba(255, 255, 255, 0.05);
    }
    .prediction-card {
        padding: 50px;
        border-radius: 30px;
        margin-top: 25px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(25px);
        transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        min-width: 400px;
        text-align: center;
        animation: fadeInUp 0.8s ease-out;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .prediction-card h3 {
        font-size: 38px !important;
        margin-bottom: 20px;
    }
    .spam { 
        border-left: 10px solid #ff4d4d; 
        background: rgba(255, 77, 77, 0.08);
        box-shadow: 0 10px 40px rgba(255, 77, 77, 0.2);
    }
    .ham { 
        border-left: 10px solid #00ff88; 
        background: rgba(0, 255, 136, 0.08);
        box-shadow: 0 10px 40px rgba(0, 255, 136, 0.2);
    }
    /* Typography */
    h1, h2, h3 { 
        color: #ffffff !important; 
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: -0.5px;
        font-weight: 700;
    }
    .stMarkdown p {
        font-family: 'Outfit', sans-serif;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# Path Resolution Helper
def resolve_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

# Load Models & Vectorizer
@st.cache_resource
def load_resources():
    try:
        tfidf_path = resolve_path("models/tfidf_vectorizer.pkl")
        nb_path = resolve_path("models/naive_bayes_model.pkl")
        lr_path = resolve_path("models/logistic_regression_model.pkl")
        svm_path = resolve_path("models/svm_model.pkl")
        metrics_path = resolve_path("models/model_metrics.pkl")
        
        vectorizer = joblib.load(tfidf_path)
        nb_model = joblib.load(nb_path)
        lr_model = joblib.load(lr_path)
        svm_model = joblib.load(svm_path)
        metrics = joblib.load(metrics_path)
        
        return vectorizer, {"Naive Bayes": nb_model, "Logistic Regression": lr_model, "SVM": svm_model}, metrics
    except Exception as e:
        st.error(f"Resource loading error: {e}")
        return None, None, None

vectorizer, models, metrics = load_resources()

# Helper: Save Prediction History
def save_history(text, label, confidence):
    history_dir = "exports"
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)
        
    history_file = os.path.join(history_dir, "prediction_history.csv")
    new_entry = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Message Snippet": text[:50] + "...",
        "Category": label,
        "Confidence": f"{confidence:.2f}%"
    }])
    if os.path.exists(history_file):
        history_df = pd.read_csv(history_file)
        history_df = pd.concat([new_entry, history_df], ignore_index=True)
    else:
        history_df = new_entry
    history_df.to_csv(history_file, index=False)

# Sidebar Navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1000/1000966.png", width=80)
    st.title("AI Spam Shield")
    page = st.radio("Navigation", ["🔍 Predict", "📊 Analytics", "📁 Bulk Process", "📜 History", "⚙️ Settings"])
    
    st.divider()
    if metrics:
        st.write("### Model Performance")
        st.info(f"🏆 Best Model: SVM\nAccuracy: {metrics['SVM']['accuracy']:.2%}")

# Predict Page
if page == "🔍 Predict":
    st.title("AI Spam Mail Detection")
    st.markdown("Enter the content of the email or message below to check if it's spam.")
    
    input_text = st.text_area("Message Content", placeholder="Write or paste your email here...", height=200)
    
    c1, c2 = st.columns([2, 3])
    with c1:
        if st.button("Analyze Message"):
            if not input_text.strip():
                st.warning("Please enter some text.")
            elif not models:
                st.error("Models not found. Please train models in Settings first.")
            else:
                with st.spinner("Analyzing with AI..."):
                    processed_text = clean_text(input_text)
                    vectorized = vectorizer.transform([processed_text])
                    
                    # Using SVM as primary (since it was best)
                    prediction = models["SVM"].predict(vectorized)[0]
                    proba = models["SVM"].predict_proba(vectorized)[0]
                    confidence = proba[prediction] * 100
                    
                    label = "SPAM" if prediction == 1 else "NOT SPAM (HAM)"
                    css_class = "spam" if prediction == 1 else "ham"
                    icon = "🚫" if prediction == 1 else "✅"
                    
                    spam_type = ""
                    if prediction == 1:
                        spam_type = categorize_spam(input_text)
                    
                    # Prepare style variables
                    accent_color = "#ff4d4d" if prediction == 1 else "#00ff88"
                    glow_color = "rgba(255,77,77,0.5)" if prediction == 1 else "rgba(0,255,136,0.5)"
                    type_badge = f'<p style="background: rgba(255,255,255,0.1); display: inline-block; padding: 8px 20px; border-radius: 12px; font-weight: 600; margin-bottom: 30px; border: 1px solid rgba(255,255,255,0.2);">Spam Type: {spam_type}</p>' if spam_type else ''
                    
                    # Construct clean HTML
                    card_html = f"""<div class="prediction-card {css_class}">
<h3 style="margin-top: 0;">{icon} {label}</h3>
{type_badge}
<p style="font-size: 24px; opacity: 0.7; margin-bottom: 0;">AI CONFIDENCE SCORE</p>
<p style="font-size: 72px; font-weight: 900; margin-top: -5px; color: {accent_color}; text-shadow: 0 0 30px {glow_color}; letter-spacing: -2px;">{confidence:.2f}%</p>
</div>"""
                    st.html(card_html)
                    
                    # Save to history
                    save_history(input_text, label, confidence)
                    
                    # Result Visualization
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = confidence,
                        title = {'text': "AI Confidence", 'font': {'size': 24}, 'align': 'center'},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                            'bar': {'color': "#ff2e63" if prediction == 1 else "#00ff88"},
                            'bgcolor': "rgba(255, 255, 255, 0.03)",
                            'borderwidth': 2,
                            'bordercolor': "rgba(255, 255, 255, 0.1)",
                            'steps' : [
                                {'range': [0, 50], 'color': "rgba(255, 255, 255, 0.02)"},
                                {'range': [50, 100], 'color': "rgba(255, 255, 255, 0.05)"}],
                        }
                    ))
                    fig.update_layout(
                        height=350,
                        margin=dict(l=40, r=40, t=100, b=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        font={'color': "white", 'family': "Outfit"},
                        autosize=True
                    )
                    st.plotly_chart(fig, use_container_width=True)

# Analytics Page
elif page == "📊 Analytics":
    st.title("Model Comparative Analysis")
    
    if metrics:
        st.markdown("Comparing different machine learning models trained on the Mail Spam dataset.")
        
        # Performance Table
        perf_data = []
        for model_name, info in metrics.items():
            perf_data.append({
                "Model": model_name,
                "Accuracy": info['accuracy'],
                "Precision": info['precision'],
                "Recall": info['recall'],
                "F1-Score": info['f1']
            })
        
        df_perf = pd.DataFrame(perf_data)
        
        col1, col2 = st.columns(2)
        with col1:
            fig_acc = px.bar(df_perf, x="Model", y="Accuracy", color="Accuracy", 
                            title="Model Accuracy Comparison", color_continuous_scale="Viridis")
            fig_acc.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            st.plotly_chart(fig_acc, use_container_width=True)
            
        with col2:
            fig_f1 = px.line(df_perf, x="Model", y=["Precision", "Recall", "F1-Score"], 
                            title="Detailed Metrics Comparison", markers=True)
            fig_f1.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            st.plotly_chart(fig_f1, use_container_width=True)
        
        st.subheader("Model Evaluation Summary")
        st.table(df_perf.style.format({
            "Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}", "F1-Score": "{:.2%}"
        }))
    else:
        st.warning("Please train models in Settings first.")

# Bulk Process Page
elif page == "📁 Bulk Process":
    st.title("Batch Spam Classification")
    st.write("Upload a CSV file containing an 'email' or 'text' column for bulk processing.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file and models:
        df_bulk = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df_bulk.head())
        
        # Intelligent Column Selection
        default_index = 0
        if 'message' in df_bulk.columns:
            default_index = list(df_bulk.columns).index('message')
        elif 'text' in df_bulk.columns:
            default_index = list(df_bulk.columns).index('text')
            
        target_col = st.selectbox("Select Text Column (where message content is)", df_bulk.columns, index=default_index)
        
        if st.button("🚀 Start Bulk Prediction"):
            # Prepare data and apply a 10,000 row limit for demo stability
            texts = df_bulk[target_col].fillna("").tolist()
            if len(texts) > 10000:
                st.warning("⚠️ Large file detected! For demo performance, we are processing the first 10,000 rows.")
                texts = texts[:10000]
                
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_rows = len(texts)
            processed_texts = []
            
            # Batch Processing Logic for Speed
            chunk_size = 500
            for i in range(0, total_rows, chunk_size):
                chunk = texts[i : i + chunk_size]
                # Process chunk
                processed_chunk = [clean_text(t) for t in chunk]
                processed_texts.extend(processed_chunk)
                
                # Update progress
                progress = min((i + chunk_size) / total_rows, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processed {min(i + chunk_size, total_rows)} of {total_rows} rows...")
                
            status_text.text("Finalizing predictions...")
            
            # Vectorization and Prediction
            vectorized = vectorizer.transform(processed_texts)
            preds = models["SVM"].predict(vectorized)
            probs = models["SVM"].predict_proba(vectorized).max(axis=1)
            
            # Create a temporary results dataframe for the limited rows
            results_df = df_bulk.iloc[:total_rows].copy()
            results_df['Prediction'] = ["SPAM" if p == 1 else "HAM" for p in preds]
            results_df['Confidence'] = [f"{p*100:.2f}%" for p in probs]
            
            # Identify Spam Types for bulk data
            spam_types = []
            for i, p in enumerate(preds):
                if p == 1:
                    spam_types.append(categorize_spam(texts[i]))
                else:
                    spam_types.append("N/A")
            results_df['Spam Type'] = spam_types
            
            progress_bar.empty()
            st.success(f"✅ Successfully processed {total_rows} rows!")
            st.dataframe(results_df[['Prediction', 'Spam Type', 'Confidence'] + [target_col]].head(10), use_container_width=True)
            
            # Download results
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Classified CSV (Limit 10k)",
                data=csv,
                file_name=f"classified_10k_{uploaded_file.name}",
                mime="text/csv"
            )

# History Page
elif page == "📜 History":
    st.title("Prediction History")
    history_file = "exports/prediction_history.csv"
    if os.path.exists(history_file):
        history_df = pd.read_csv(history_file)
        st.dataframe(history_df, use_container_width=True)
        if st.button("Clear History"):
            os.remove(history_file)
            st.rerun()
    else:
        st.info("No prediction history found.")

# Settings Page
elif page == "⚙️ Settings":
    st.title("System Configuration")
    st.subheader("Model Retraining")
    st.write("You can retrain the models if the dataset has been updated.")
    
    if st.button("🔥 Re-train All Models"):
        with st.spinner("Retraining... Please wait (this may take a minute)"):
            results = train_models()
            if results:
                st.success("Models retrained successfully!")
                st.balloons()
                time.sleep(2)
                st.rerun()

st.divider()
