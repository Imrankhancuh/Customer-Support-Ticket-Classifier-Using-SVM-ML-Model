import nltk
import os

# ===============================
# DOWNLOAD NLTK DATA LOCALLY
# ===============================
NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)

nltk.data.path.append(NLTK_DATA_DIR)

# Required packages only
packages = [
    "stopwords",
    "wordnet",
    "omw-1.4"
]

for pkg in packages:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg, download_dir=NLTK_DATA_DIR)

print("✅ NLTK setup completed")
