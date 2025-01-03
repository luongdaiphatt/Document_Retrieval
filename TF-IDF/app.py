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

# TF_IDF
def TF_IDF_search(query, k=50):
    vectorizer = TfidfVectorizer()
    corpus_tfidf = vectorizer.fit_transform(tokenized_corpus + [query])
    cosine_similarities = cosine_similarity(corpus_tfidf[-1], corpus_tfidf[:-1])[0]
    top_indices = cosine_similarities.argsort()[::-1][:k]
    top_similarities = cosine_similarities[top_indices]
    return top_indices.tolist(), top_similarities.tolist()

# Mô phỏng tìm kiếm
@app.route("/")
def home():
    return render_template("news/home.html")

@app.route("/search", methods=["POST", "GET"])
def submit():
    global query, top_indices_tf_idf, tf_idf_scores
    if request.method == "POST":
        query = request.form['search']
        p_query = preprocess_text(query)

        ### TF-IDF
        top_indices_tf_idf, tf_idf_scores = TF_IDF_search(p_query, 50)

    # TF-IDF: ưu tiên những từ khóa đơn lẻ có số lần xuất hiện nhiều + thứ tự các từ query không ảnh hưởng
    page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')
    pagination = Pagination(page=page, per_page=per_page, total=len(top_indices_tf_idf), css_framework='bootstrap5')
    k_idx_show = top_indices_tf_idf[offset: offset + NUM_NEWS_PER_PAGE]
    return render_template("news/search.html", k_idx=k_idx_show, data=data, similarities=tf_idf_scores, query=query, pagination=pagination)


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

    # Download SBERT (save model if not exists)
    sbert_model_dir = os.path.join(parent_dir, 'sbert_model')
    if not os.path.exists(sbert_model_dir):
        sbert = SentenceTransformer('keepitreal/vietnamese-sbert')
        sbert.save(sbert_model_dir)
        print("SBERT downloaded")
    else:
        sbert = SentenceTransformer(sbert_model_dir)
        print("SBERT loaded")

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

    app.run(debug=True)