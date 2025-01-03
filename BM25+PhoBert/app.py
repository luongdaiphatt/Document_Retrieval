from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from flask_paginate import Pagination, get_page_args
from flask import Flask, request, render_template
from transformers import AutoModel, AutoTokenizer
from rank_bm25 import BM25Okapi
import py_vncorenlp
import numpy as np
import pickle
import torch
import json
import sys
import os

app = Flask(__name__)

# thêm đường dẫn chung cho các model
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)
data_dir = os.path.join(parent_dir, 'data')

# Các biến toàn cục
NUM_NEWS_PER_PAGE = 10
items = list()
query = ""
similarities = list()
top_indices_tf_idf = list()

special_characters = [
    '!', '"', '#', '$', '%', '&', "'", '(', ')', '*',
    '+', ',', '-', '.', '/', ':', ';', '<', '=', '>',
    '?', '@', '[', '\\', ']', '^', '`', '{', '|',
    '}', '~'
]

# Tiền xử lý văn bản
def lower_text(text):
    ptext = text.lower()
    return ptext

def segment_text(text):
    text = rdrsegmenter.word_segment(text)
    return str(text)

def remove_stopwords(stopwords, title):
    tokenized_title = title.split(' ')
    fi_title_words = []
    for word in tokenized_title:
        if word not in stopwords:
            fi_title_words.append(word)
    fi_title = ' '.join(fi_title_words)
    return fi_title

def remove_special_characters(title):
    cleaned_title = ''.join(char for char in title if char not in special_characters)
    return cleaned_title

def preprocess_text(text):
    ptext = remove_special_characters(remove_stopwords(stopwords, segment_text(lower_text(text))))
    return ptext

def encode_phobert(corpus, max_length=256, batch_size=8):
    encoded_corpus = []
    phobert.to(device)

    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i + batch_size]
        encoded_inputs = tokenizer(
            batch, max_length=max_length, truncation=True, padding=True, return_tensors="pt"
        )
        input_ids = encoded_inputs["input_ids"].to(device)
        attention_mask = encoded_inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = phobert(input_ids, attention_mask=attention_mask)
            embeddings = outputs.pooler_output.cpu().numpy()

        encoded_corpus.extend(embeddings)

    return np.array(encoded_corpus)

# BM25 và PhoBERT tìm kiếm
def bm25_search(query, k=50):
    split_query = query.split()
    scores = bm25.get_scores(split_query)
    top_indices = scores.argsort()[::-1][:k]
    top_similarities = scores[top_indices]
    return top_indices.tolist(), top_similarities.tolist()

def phobert_search(top_indices_bm25, query, k=50):
    query_ids = tokenizer.encode(query, max_length=256, truncation=True, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        encoded_query = phobert(query_ids).pooler_output.cpu().numpy().flatten()
    similarities = np.dot(encoded_corpus_phobert[top_indices_bm25], encoded_query) / (
        np.linalg.norm(encoded_corpus_phobert[top_indices_bm25], axis=1) * np.linalg.norm(encoded_query)
    )
    top_indices = similarities.argsort()[::-1][:k]
    top_similarities = similarities[top_indices]
    # map the indices back to the original indices (BM25)
    top_indices = [top_indices_bm25[i] for i in top_indices]
    return top_indices, top_similarities.tolist()

# Mô phỏng tìm kiếm
@app.route("/")
def home():
    return render_template("news/home.html")

@app.route("/search", methods=["POST", "GET"])
def submit():
    global query, top_indices_tf_idf, tf_idf_scores, top_indices_sbert, sbert_scores, top_indices_bm25, bm25_scores, top_indices_phobert, phobert_scores
    if request.method == "POST":
        query = request.form['search']
        p_query = preprocess_text(query)

        ### BM25
        top_indices_bm25, bm25_scores = bm25_search(p_query, 50)
        ### BM25 + PhoBERT
        top_indices_phobert, phobert_scores = phobert_search(top_indices_bm25, p_query)

    ### BM25 + PhoBERT: tương tự như TF-IDF + SBERT nhưng sử dụng PhoBERT thay vì SBERT (thỉnh thoảng cho kết quả không liên quan như khi search "cá lóc")
    page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')
    pagination = Pagination(page=page, per_page=per_page, total=len(top_indices_phobert), css_framework='bootstrap5')
    k_idx_show = top_indices_phobert[offset: offset + NUM_NEWS_PER_PAGE]
    return render_template("news/search.html", k_idx=k_idx_show, data=data, similarities=phobert_scores, query=query, pagination=pagination)


if __name__ == "__main__":

    rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=parent_dir)

    stopwords_path = os.path.join(data_dir, 'vietnamese-stopwords-dash.txt')
    if os.path.exists(stopwords_path):
        stopwords = open(stopwords_path, 'r',
                         encoding='utf-8').read().split("\n")
    else:
        print(f"File not found: {stopwords_path}")
        stopwords = []

    data_path = os.path.join(data_dir, 'ArticlesNewspaper.json')
    if os.path.exists(data_path):
        data = json.load(open(data_path, 'r', encoding="utf-8"))
    else:
        print(f"File not found: {data_path}")
        data = []

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Save preprocessed data if not exists
    tokenized_corpus_path = os.path.join(data_dir, 'tokenized_corpus.txt')
    if not os.path.exists(tokenized_corpus_path):
        data_text = [i['title'] + " " + i['abstract'] for i in data]
        tokenized_corpus = [preprocess_text(corpus) for corpus in data_text]
        with open(tokenized_corpus_path, 'w', encoding='utf-8') as f:
            for item in tokenized_corpus:
                f.write("%s\n" % item)
        print("Saved data")
    else:
        tokenized_corpus = open(tokenized_corpus_path,
                                'r', encoding='utf-8').read().split("\n")
        print("Data loaded")

    # for BM25 search
    # BM25 Training (save trained model if not exists)
    bm25_model_path = os.path.join(data_dir, 'bm25_model.pkl')
    if not os.path.exists(bm25_model_path):
        split_tokenized_corpus = [doc.split() for doc in tokenized_corpus]
        bm25 = BM25Okapi(split_tokenized_corpus)
        # Save the BM25 model
        with open(bm25_model_path, 'wb') as bm25_file:
            pickle.dump(bm25, bm25_file)
        print("BM25 saved")
    else:
        with open(bm25_model_path, 'rb') as bm25_file:
            bm25 = pickle.load(bm25_file)
        print("BM25 loaded")

    # Download PhoBERT (save model if not exists)
    phobert_model_dir = os.path.join(parent_dir, 'phobert_model')
    if not os.path.exists(phobert_model_dir):
        phobert = AutoModel.from_pretrained(
            "vinai/phobert-base-v2", add_pooling_layer=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        phobert.save_pretrained(phobert_model_dir)
        tokenizer.save_pretrained(phobert_model_dir)
        print("PhoBERT downloaded")
    else:
        phobert = AutoModel.from_pretrained(phobert_model_dir).to(device)
        tokenizer = AutoTokenizer.from_pretrained(phobert_model_dir)
        print("PhoBERT loaded")

    # for PhoBERT search
    # Save the encoded corpus if not exists
    encoded_corpus_phobert_path = os.path.join(
        data_dir, 'encoded_corpus_phobert.npy')
    if not os.path.exists(encoded_corpus_phobert_path):
        encoded_corpus_phobert = encode_phobert(tokenized_corpus)
        np.save(encoded_corpus_phobert_path, encoded_corpus_phobert)
        print("PhoBERT encoded corpus saved")
    else:
        encoded_corpus_phobert = np.load(encoded_corpus_phobert_path)
        print("PhoBERT encoded corpus loaded")

    app.run(debug=True)