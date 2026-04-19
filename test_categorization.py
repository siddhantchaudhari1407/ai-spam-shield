import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.spam_engine import categorize_spam

test_cases = [
    ("Congratulations! You won a £1000 prize. Claim now.", "Lottery & Prize Scam"),
    ("URGENT: Your bank account has been suspended. Verify here.", "Financial & Phishing"),
    ("Get 50% discount on all items. Shop today!", "Marketing & Promo"),
    ("Hi, I'm Randy. Wanna meet up for some fun?", "Adult & Dating"),
    ("Urgent hiring for online part-time jobs. Earn $500 daily.", "Job & Recruitment Scam"),
    ("Buy 1 get 1 free on all shoes.", "Marketing & Promo"),
    ("Official notice: Please login to update security.", "Financial & Phishing"),
    ("Just a random message about the weather.", "General Spam")
]

print("Starting Categorization Tests...")
passed = 0
for text, expected in test_cases:
    result = categorize_spam(text)
    if result == expected:
        print(f"PASS: '{text[:20]}...' -> {result}")
        passed += 1
    else:
        print(f"FAIL: '{text[:20]}...' -> Got {result}, Expected {expected}")

print(f"\nResult: {passed}/{len(test_cases)} tests passed.")
