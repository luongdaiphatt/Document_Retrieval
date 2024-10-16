import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pyvi import ViTokenizer

class DocumentRetrieval:
    NUM_NEWS_PER_PAGE = 12

    def __init__(self):
        self.items = []
        self.similarities = []
        self.df = pd.DataFrame()
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = []

    def preprocess_stopwords(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            stopwords = [ViTokenizer.tokenize(line.strip().lower().replace(' ', '_')) for line in f]
        return stopwords

    def segment_text(self, text):
        return ViTokenizer.tokenize(text)

    def unsegment_text(self, text):
        text = text.replace('_', ' ').replace(' ,', ',').replace(' .', '.').replace('( ', '(')
        text = text.replace(' )', ')').replace(' :', ':').replace(' ;', ';').replace(' ?', '?')
        text = text.replace(' !', '!').replace('“ ', '“').replace(' ”', '”')
        text = text.replace("' ", "'").replace(" '", "'").replace(' / ', '/')
        return text
    
    def resize_image(self, image_url):
        if image_url.find("w=300&h=186") != -1:
            return image_url.replace("w=300&h=186", "w=224&h=163")
        # elif image_url.find("w=300&h=180") != -1: # vnexpress not working
        #     return image_url.replace("w=300&h=180", "w=224&h=163")
        elif image_url.find("w=22&h=14") != -1: # Lao Dong 
            return image_url.replace("w=22&h=14", "w=224&h=163")
        # elif image_url.find("378_252") != -1: # dantri not working
        #     return image_url.replace("378_252", "224_163")
        return image_url

    def init(self, csv_file, stopwords_file):
        # Load the CSV file into a DataFrame
        self.df = pd.read_csv(csv_file)
        self.df['title'] = self.df['title'].apply(self.segment_text)
        self.df['abstract'] = self.df['abstract'].apply(self.segment_text)
        data_text = self.df['title'] + " " + self.df['abstract']
        
        # Load and preprocess stopwords
        stopwords = self.preprocess_stopwords(stopwords_file)
        
        # Initialize TF-IDF Vectorizer with preprocessed stop words
        self.vectorizer = TfidfVectorizer(stop_words=stopwords)
        
        # Fit and transform the data
        self.tfidf_matrix = self.vectorizer.fit_transform(data_text)

    def search(self, query, page_number=1):
        query_segmented = self.segment_text(query)
        query_tfidf = self.vectorizer.transform([query_segmented])
        self.similarities = cosine_similarity(
            query_tfidf, self.tfidf_matrix).flatten()

        # Sort by similarity and get the indices
        sorted_indices = self.similarities.argsort()[::-1]

        # Calculate pagination
        start_idx = (page_number - 1) * self.NUM_NEWS_PER_PAGE
        end_idx = start_idx + self.NUM_NEWS_PER_PAGE

        # Get the items for the requested page
        page_indices = sorted_indices[start_idx:end_idx]
        # Use .copy() to avoid SettingWithCopyWarning
        self.items = self.df.iloc[page_indices].copy()

        # Unsegment the title and abstract
        self.items.loc[:, 'title'] = self.items['title'].apply(
            self.unsegment_text)
        self.items.loc[:, 'abstract'] = self.items['abstract'].apply(
            self.unsegment_text)
        self.items.loc[:, 'imglink'] = self.items['imglink'].apply(
            self.resize_image
        )

        # Convert DataFrame to JSON-serializable format
        results = self.items.to_dict(orient='records')

        return results, self.similarities[page_indices].tolist()

# Example usage
if __name__ == "__main__":
    dr = DocumentRetrieval()
    dr.init('data/crawled.csv', 'data/vietnamese-stopwords.txt')
    results, similarities = dr.search("điện thoại", page_number=1)
    print(results)
    print(similarities)