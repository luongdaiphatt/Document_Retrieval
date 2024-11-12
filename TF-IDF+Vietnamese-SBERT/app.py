from flask import Flask, request, render_template
from flask_paginate import Pagination, get_page_args
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
import py_vncorenlp
import json
import numpy as np

app = Flask(__name__)

NUM_NEWS_PER_PAGE = 10
items = list()
query = ""
similarities=list()
top_indices = list()

special_characters = [
    '!', '"', '#', '$', '%', '&', "'", '(', ')', '*',
    '+', ',', '-', '.', '/', ':', ';', '<', '=', '>',
    '?', '@', '[', '\\', ']', '^', '`', '{', '|',
    '}', '~'
]
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir= r'C:\Users\Admin\AppData\Local\Programs\Python\Python311\Lib\site-packages\py_vncorenlp')
os.chdir(r'C:\Users\Admin\Desktop\New folder (13)\VNnSE_Flask\TF-IDF+Vietnamese-SBERT')

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

def TF_IDF(query):
    vectorizer = TfidfVectorizer()
    corpus_tfidf = vectorizer.fit_transform(data_text + [query])
    cosine_similarities = cosine_similarity(corpus_tfidf[-1], corpus_tfidf[:-1])
    top_indices = cosine_similarities.argsort()[0][::-1]
    return top_indices

def SBERT(top_indices, query):
    global items
    corpus = [data_text[top_indices[i]] for i in range(50)]
    prev_index = [top_indices[i] for i in range(50)]
    model = SentenceTransformer('keepitreal/vietnamese-sbert')
    encoded_corpus = model.encode(corpus)
    encoded_query = model.encode(query)
    cossimilarity = []
    for i in range(encoded_corpus.shape[0]):
        cossimilarity.append(encoded_corpus[i, :].dot(encoded_query) / (np.linalg.norm(encoded_corpus[i, :]) * np.linalg.norm(encoded_query)))
    sorted_ids_data = np.argsort(np.array(cossimilarity))
    sorted_ids = sorted_ids_data[::-1]
    k = len(corpus)
    k_idx = sorted_ids[0: k]
    items = k_idx
    return k_idx, cossimilarity

@app.route("/")
def home():
    return render_template("news/home.html")

@app.route("/search", methods=["POST", "GET"])
def submit():
    global items, query, similarities, top_indices
    if request.method == "POST":
        query = request.form['search']
        p_query = remove_special_characters(remove_stopwords(stopwords, segment_text(lower_text(query))))
        top_indices = TF_IDF(p_query)
        k_idx, similarities = SBERT(top_indices, query)
    page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')
    pagination = Pagination(page=page, per_page=per_page, total=len(items), css_framework='bootstrap5')
    k_idx_show = items[offset: offset + NUM_NEWS_PER_PAGE]
    return render_template("news/search.html", k_idx=k_idx_show, data=data, similarities=similarities, query=query, pagination=pagination, index=top_indices)


if __name__ == "__main__":
    data = json.load(open(r'data\ArticlesNewspaper.json', 'r', encoding="utf-8"))
    data_text = [i['title'] + " " + i['abstract'] for i in data]
    titles = [i['title'] for i in data]
    stopwords = open(r"data\vietnamese-stopwords-dash.txt",'r',encoding='utf-8').read().split("\n")
    prpfi = open(r"data\preprocessing.txt", 'r', encoding='utf-8').read().split("\n")
    #test_query = remove_special_characters(remove_stopwords(stopwords, segment_text(lower_text("Ronaldo giàu cỡ nào?"))))
    tokenized_corpus = [doc.split(" ") for doc in prpfi]
    app.run(debug=True)