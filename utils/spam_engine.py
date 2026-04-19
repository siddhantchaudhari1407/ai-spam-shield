import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure NLTK data is downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Pre-initialize resources for performance
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def categorize_spam(text):
    """
    Identifies the type of spam based on content keywords.
    """
    text = text.lower()
    categories = {
        "Job & Recruitment Scam": ["job", "hiring", "salary", "interview", "position", "recruit", "work from home", "income", "part-time"],
        "Lottery & Prize Scam": ["win", "won", "prize", "claim", "cash", "award", "jackpot", "winner", "lottery"],
        "Financial & Phishing": ["urgent", "verify", "security", "alert", "bank", "account", "suspended", "login", "official"],
        "Marketing & Promo": ["free", "offer", "discount", "voucher", "sale", "deal", "subscribe", "gift", "stop"],
        "Adult & Dating": ["dating", "sexy", "meet", "adult", "hot", "flirt", "lonely", "contact"]
    }
    for category, keywords in categories.items():
        if any(word in text for word in keywords):
            return category
    return "General Spam"

def clean_text(text):
    """
    NLP preprocessing pipeline optimized for batch performance:
    - Lowercase
    - Remove punctuation
    - Remove numbers
    - Remove stopwords
    - Stemming
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation & numbers
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    
    # Tokenization and Stopword removal
    words = text.split()
    cleaned_words = [ps.stem(w) for w in words if w not in stop_words]
    
    return " ".join(cleaned_words)
