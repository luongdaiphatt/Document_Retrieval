import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Constants
NUM_NEWS_PER_PAGE = 5
items = []
similarities = []

# Function to process query and return a TF-IDF vector for it
def query_input(query):
    query_vector = vectorizer.transform([query])
    return query_vector

# Function to calculate cosine similarity between query and all articles
def cos_similarity(query_vector):
    global items
    similarities = cosine_similarity(query_vector, Tf_idf_matrix).flatten()
    sorted_ids = similarities.argsort()[::-1]  # Sort by similarity, highest first
    k = 100  # Number of top results to show
    k_idx = sorted_ids[0:k]  # Get indices of top-k results
    items = k_idx
    return similarities.astype(str)

# Function to paginate results
def paginate(items, page, per_page):
    start = (page - 1) * per_page
    end = start + per_page
    return items[start:end]

# Main function to run the search and display results
def search(query, page=1):
    global similarities
    query_vector = query_input(query)
    similarities = cos_similarity(query_vector)

    # Paginate the results
    paginated_items = paginate(items, page, NUM_NEWS_PER_PAGE)

    # Show the results
    for idx in paginated_items:
        print(f"Title: {df.iloc[idx]['title']}")
        print(f"Abstract: {df.iloc[idx]['abstract']}")
        print(f"Similarity Score: {similarities[idx]}")
        print('-' * 40)

if __name__ == "__main__":
    # Load the CSV file into a DataFrame
    df = pd.read_csv('data/vtc.csv')

    # Combine 'title' and 'abstract' columns for TF-IDF
    data_text = df['title'] + " " + df['abstract']

    # Load stopwords (optional, assuming you have a file of stopwords)
    stopwords = open('data/vietnamese-stopwords.txt', 'r', encoding='utf-8').read().split("\n")

    # Create TF-IDF vectorizer and fit it to the dataset
    vectorizer = TfidfVectorizer(stop_words=stopwords)
    Tf_idf_matrix = vectorizer.fit_transform(data_text)

    # Example usage
    query = input("Enter your search query: ")
    page = int(input("Enter the page number to display: "))
    
    search(query, page)
