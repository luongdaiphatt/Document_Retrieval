from rank_bm25 import BM25Okapi
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import json
import py_vncorenlp
import os
import pickle

NUM_RESULTS = 10
special_characters = [
    '!', '"', '#', '$', '%', '&', "'", '(', ')', '*',
    '+', ',', '-', '.', '/', ':', ';', '<', '=', '>',
    '?', '@', '[', '\\', ']', '^', '`', '{', '|',
    '}', '~'
]
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], 

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tải PhoBERT
phobert = AutoModel.from_pretrained("vinai/phobert-base-v2").to(device)
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

# 1. Tiền xử lý văn bản
def lower_text(text):
    return text.lower()

def segment_text(text):
    return str(rdrsegmenter.word_segment(text))

def remove_stopwords(stopwords, text):
    tokenized_text = text.split(' ')
    return ' '.join([word for word in tokenized_text if word not in stopwords])

def remove_special_characters(text):
    return ''.join(char for char in text if char not in special_characters)

# 2. Load dữ liệu và tiền xử lý
data = json.load(open('data/ArticlesNewspaper.json', 'r', encoding="utf-8"))
data_text = [item['title'] + " " + item['abstract'] for item in data]
stopwords = open('data/vietnamese-stopwords-dash.txt', 'r', encoding='utf-8').read().split("\n")
tokenized_corpus = [remove_special_characters(remove_stopwords(stopwords,segment_text(lower_text(doc)))) for doc in data_text]

# BM25 Training (save trained model if not exists)
if not os.path.exists('bm25_model.pkl'): 
    bm25 = BM25Okapi(tokenized_corpus)
    # Save the BM25 model
    with open('bm25_model.pkl', 'wb') as bm25_file:
        pickle.dump(bm25, bm25_file)
else:
    with open('bm25_model.pkl', 'rb') as bm25_file:
        bm25 = pickle.load(bm25_file)

# PhoBERT Encoding
def encode_phobert(corpus, max_length=256):
    encoded_corpus = []
    for text in corpus:
        # Truncate the text to the maximum length
        input_ids = torch.tensor([tokenizer.encode(text, max_length=max_length, truncation=True)]).to(device)
        with torch.no_grad():
            output = phobert(input_ids).pooler_output.cpu().numpy().flatten()
        encoded_corpus.append(output)
    return np.array(encoded_corpus)

encoded_corpus_phobert = encode_phobert(data_text)

# 3. BM25 và PhoBERT tìm kiếm
def bm25_search(query):
    tokenized_query = segment_text(lower_text(query)).split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:50]
    return top_indices, scores[top_indices]

def phobert_search(query, top_indices):
    query_ids = torch.tensor([tokenizer.encode(query)]).to(device)
    with torch.no_grad():
        encoded_query = phobert(query_ids).pooler_output.cpu().numpy().flatten()
    similarities = np.dot(encoded_corpus_phobert[top_indices], encoded_query) / (
        np.linalg.norm(encoded_corpus_phobert[top_indices], axis=1) * np.linalg.norm(encoded_query)
    )
    sorted_indices = np.argsort(similarities)[::-1]
    return sorted_indices, similarities[sorted_indices]

query = "Cách ngăn chặn dịch bệnh"  
processed_query = remove_special_characters(remove_stopwords(stopwords, segment_text(lower_text(query))))
bm25_indices, bm25_scores = bm25_search(processed_query)
phobert_sorted_indices, phobert_similarities = phobert_search(processed_query, bm25_indices)

print("Top kết quả tìm kiếm dựa trên PhoBERT:")
for rank, idx in enumerate(phobert_sorted_indices[:NUM_RESULTS]):
    print(f"{rank + 1}. {data[bm25_indices[idx]]['title']} (BM25 Score: {bm25_scores[idx]:.4f}, PhoBERT Score: {phobert_similarities[rank]:.4f})")

print("Top kết quả tìm kiếm dựa trên BM25:")
for rank, idx in enumerate(bm25_indices[:NUM_RESULTS]):
    print(f"{rank + 1}. {data[idx]['title']} (BM25 Score: {bm25_scores[rank]:.4f}, PhoBERT Score: {phobert_similarities[rank]:.4f})")