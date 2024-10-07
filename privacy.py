import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the original dataset and take the first 50 entries
original_data = pd.read_csv('real.csv').head(50)

# Load synthetic datasets
synthetic_data_chatgpt = pd.read_csv('Data/chatgpt.csv')
synthetic_data_llama = pd.read_csv('Data/llama2.csv')
synthetic_data_claude = pd.read_csv('Data/claude.csv')

# Combine synthetic datasets into a dictionary for easier processing
synthetic_datasets = {
    'ChatGPT': synthetic_data_chatgpt,
    'Llama': synthetic_data_llama,
    'Claude': synthetic_data_claude
}

# Function to preprocess and vectorize text data using a given vectorizer
def preprocess_and_vectorize(data, text_fields, vectorizer=None):
    combined_text = data[text_fields].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    if vectorizer is None:
        vectorizer = TfidfVectorizer().fit(combined_text)
    tfidf_matrix = vectorizer.transform(combined_text)
    return tfidf_matrix, vectorizer

# Define the text fields to compare
text_fields = ['rating', 'title', 'text', 'asin', 'parent_asin', 'user_id', 'timestamp']

# Preprocess and vectorize original data
original_tfidf_matrix, original_vectorizer = preprocess_and_vectorize(original_data, text_fields)

# Function to check for leaks in synthetic data
def check_for_leaks(original_matrix, vectorizer, synthetic_data, text_fields):
    synthetic_tfidf_matrix, _ = preprocess_and_vectorize(synthetic_data, text_fields, vectorizer)
    similarity_matrix = cosine_similarity(synthetic_tfidf_matrix, original_matrix)
    leaks = similarity_matrix.max(axis=1)
    return leaks

# Evaluate each synthetic dataset
for model_name, synthetic_data in synthetic_datasets.items():
    leaks = check_for_leaks(original_tfidf_matrix, original_vectorizer, synthetic_data, text_fields)
    leak_summary = leaks > 0.8  # Threshold for considering a text as leaked
    print(f"Leakage summary for {model_name}:")
    print(f"Number of leaked entries: {leak_summary.sum()}")
    print(f"Percentage of leaked entries: {leak_summary.mean() * 100:.2f}%")
    print()
