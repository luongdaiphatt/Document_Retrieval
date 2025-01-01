from flask import Flask, request, render_template
from flask_paginate import Pagination, get_page_args
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
import json
import numpy as np
import py_vncorenlp

app = Flask(__name__)

NUM_NEWS_PER_PAGE = 10
special_characters = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"])

def preprocess_text(text, stopwords):
    text = ''.join(char for char in text.lower() if char not in special_characters)
    segmented_text = rdrsegmenter.word_segment(text)
    return ' '.join([word for word in segmented_text.split() if word not in stopwords])

def calculate_tfidf_similarity(query, corpus):
    vectorizer = TfidfVectorizer()
    corpus_tfidf = vectorizer.fit_transform(corpus + [query])
    cosine_similarities = cosine_similarity(corpus_tfidf[-1], corpus_tfidf[:-1])
    top_indices = cosine_similarities.argsort()[0][::-1]
    return top_indices

def calculate_sbert_similarity(top_indices, query, model):
    corpus = [data_text[idx] for idx in top_indices[:50]]
    encoded_corpus = model.encode(corpus)
    encoded_query = model.encode(query)
    cosine_similarities = cosine_similarity([encoded_query], encoded_corpus)[0]
    sorted_ids = np.argsort(cosine_similarities)[::-1]
    return [top_indices[idx] for idx in sorted_ids[:len(corpus)]], cosine_similarities[sorted_ids]

@app.route("/")
def home():
    return render_template("news/home.html")

@app.route("/search", methods=["POST", "GET"])
def submit():
    if request.method == "POST":
        query = request.form['search']
        processed_query = preprocess_text(query, stopwords)
        top_indices = calculate_tfidf_similarity(processed_query, data_text)
        k_idx, similarities = calculate_sbert_similarity(top_indices, query, model)
    else:
        k_idx, similarities = [], []
    
    page, per_page, offset = get_page_args(per_page=NUM_NEWS_PER_PAGE)
    pagination = Pagination(page=page, per_page=per_page, total=len(k_idx), css_framework='bootstrap5')
    k_idx_show = k_idx[offset: offset + per_page]
    
    return render_template("news/search.html", k_idx=k_idx_show, data=data, similarities=similarities, query=query, pagination=pagination)

if __name__ == "__main__":
    data_file = os.getenv('DATA_FILE', 'data/ArticlesNewspaper.json')
    data = json.load(open(data_file, 'r', encoding="utf-8"))
    data_text = [f"{item.get('title', '')} {item.get('abstract', '')}" for item in data]
    stopwords_file = os.getenv('STOPWORDS_FILE', 'data/vietnamese-stopwords-dash.txt')
    stopwords = set(open(stopwords_file, 'r', encoding='utf-8').read().splitlines())
    model = SentenceTransformer('keepitreal/vietnamese-sbert')
    app.run(debug=True)
