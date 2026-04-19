import requests
import pandas as pd
import io
import os

URL = "https://raw.githubusercontent.com/mohitgupta-1O1/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
DATA_DIR = "data"
FILE_PATH = os.path.join(DATA_DIR, "spam.csv")

def download_dataset():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    print(f"Downloading dataset from {URL}...")
    try:
        response = requests.get(URL)
        response.raise_for_status()
        
        # The file is encoded in ISO-8859-1 (latin-1) many times for this dataset
        df = pd.read_csv(io.BytesIO(response.content), encoding='latin-1')
        
        # Cleanup: Drop unnecessary columns
        df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')
        df.columns = ['label', 'text']
        
        df.to_csv(FILE_PATH, index=False)
        print(f"Dataset saved to {FILE_PATH}")
        return df
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

if __name__ == "__main__":
    download_dataset()
