import os
import re
import json
import pandas as pd
import nltk
from collections import Counter

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

import gradio as gr

# ===============================
# LOAD DATASET
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
excel_files = [f for f in os.listdir(BASE_DIR) if f.lower().endswith((".xlsx", ".xls"))]

if not excel_files:
    raise FileNotFoundError("❌ No Excel file found")

FILE_PATH = os.path.join(BASE_DIR, excel_files[0])
df = pd.read_excel(FILE_PATH)
df.fillna("", inplace=True)

# ===============================
# REQUIRED COLUMNS CHECK
# ===============================
required_cols = ["ticket_text", "issue_type", "urgency_level"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"❌ Missing column: {col}")

PRODUCT_COL = None
for col in ["product", "product_name", "item_name"]:
    if col in df.columns:
        PRODUCT_COL = col
        break

if PRODUCT_COL is None:
    raise ValueError("❌ No product column found")

# ===============================
# TEXT CLEANING
# ===============================
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = nltk.word_tokenize(text)
    return " ".join(lemmatizer.lemmatize(t) for t in tokens if t not in stop_words)

df["clean_text"] = df["ticket_text"].apply(clean_text)

# ===============================
# URGENCY NORMALIZATION
# ===============================
df["urgency_level"] = (
    df["urgency_level"]
    .astype(str)
    .str.lower()
    .map({"low": "Low", "medium": "Medium", "high": "High", "": ""})
)

# ===============================
# TRAINING DATA
# ===============================
X_issue = df["clean_text"]
y_issue = df["issue_type"]

X_urg = df["clean_text"]
y_urg = df["urgency_level"]

df["product_text"] = df["issue_type"] + " " + df["clean_text"]
X_product = df["product_text"]
y_product = df[PRODUCT_COL].astype(str)

# ===============================
# ML MODELS
# ===============================
issue_model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ("svm", LinearSVC(class_weight="balanced"))
])
issue_model.fit(X_issue, y_issue)

urgency_model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=4000)),
    ("svm", LinearSVC(class_weight="balanced"))
])
urgency_model.fit(X_urg, y_urg)

product_model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 3))),
    ("svm", LinearSVC(class_weight="balanced"))
])
product_model.fit(X_product, y_product)

# ===============================
# PRODUCT PRIOR
# ===============================
product_prior = (
    df.groupby("issue_type")[PRODUCT_COL]
    .agg(lambda x: x.value_counts().idxmax())
    .to_dict()
)

# ===============================
# DATASET SIMILARITY MATCHING
# ===============================
tfidf_matcher = TfidfVectorizer(max_features=8000)
ticket_matrix = tfidf_matcher.fit_transform(df["clean_text"])

def match_from_dataset(ticket_text, column, threshold=0.55):
    clean = clean_text(ticket_text)
    vec = tfidf_matcher.transform([clean])
    sims = cosine_similarity(vec, ticket_matrix)[0]
    idx = sims.argmax()

    if sims[idx] >= threshold:
        value = df.iloc[idx][column]
        return value if value != "" else ""

    return None

# ===============================
# ENTITY EXTRACTION
# ===============================
MONTHS = (
    "january|february|march|april|may|june|july|august|"
    "september|october|november|december"
)

COMPLAINT_KEYWORDS = [
    "urgent", "asap", "immediately",
    "failed", "error", "down",
    "not working", "broken", "damaged",
    "delay", "late", "missing",
    "refund", "cancel", "complaint",
    "payment issue", "payment failed", "transaction failed"
]

issue_keywords = {
    clean_text(i).replace("_", " ")
    for i in df["issue_type"].unique()
    if i
}

ticket_word_counts = Counter(
    word
    for text in df["clean_text"]
    for word in text.split()
    if len(word) > 4
)

frequent_ticket_keywords = {
    word for word, count in ticket_word_counts.items() if count >= 10
}

ALL_COMPLAINT_KEYWORDS = set(COMPLAINT_KEYWORDS) | issue_keywords | frequent_ticket_keywords

def extract_dates(text):
    text = text.lower()
    patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        rf"\b\d{{1,2}}\s(?:{MONTHS})(?:\s\d{{4}})?\b"
    ]
    return list({m.group() for p in patterns for m in re.finditer(p, text)})

def extract_complaints(text):
    text = text.lower()
    matches = []

    for kw in ALL_COMPLAINT_KEYWORDS:
        if kw in text:
            matches.append(kw)

    if not matches:
        return []

    matches.sort(key=lambda k: (-len(k.split()), text.find(k)))
    return [matches[0]]

# ===============================
# RULE-BASED URGENCY
# ===============================
HIGH_RULES = [
    "urgent", "asap", "immediately",
    "not working", "failed", "error", "broken"
]

def rule_based_urgency(text):
    matched = match_from_dataset(text, "urgency_level")
    if matched is not None:
        return matched

    if any(k in text.lower() for k in HIGH_RULES):
        return "High"

    return urgency_model.predict([clean_text(text)])[0]

# ===============================
# PRODUCT PREDICTION
# ===============================
def predict_product(ticket_text, issue_type):
    matched = match_from_dataset(ticket_text, PRODUCT_COL)
    if matched:
        return matched

    clean = clean_text(ticket_text)
    combined = issue_type + " " + clean

    decision = product_model.named_steps["svm"].decision_function(
        product_model.named_steps["tfidf"].transform([combined])
    )

    if decision.max() < 0.15:
        return product_prior.get(issue_type, "Unknown")

    return product_model.predict([combined])[0]

# ===============================
# FINAL CLASSIFIER
# ===============================
def classify_ticket(ticket_text):
    issue_matched = match_from_dataset(ticket_text, "issue_type")
    issue = issue_matched if issue_matched is not None else issue_model.predict([clean_text(ticket_text)])[0]

    urgency = rule_based_urgency(ticket_text)

    return {
        "issue_type": issue,
        "urgency_level": urgency,
        "product": predict_product(ticket_text, issue),
        "entities": {
            "dates": extract_dates(ticket_text),
            "complaint_keywords": extract_complaints(ticket_text)
        }
    }

# ===============================
# GRADIO UI
# ===============================
iface = gr.Interface(
    fn=lambda x: json.dumps(classify_ticket(x), indent=4),
    inputs=gr.Textbox(lines=5, label="Ticket Text"),
    outputs=gr.Textbox(lines=20, label="Prediction"),
    title="Customer Support Ticket Classifier Using SVM ML Model"
)

if __name__ == "__main__":
    iface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False
    )

