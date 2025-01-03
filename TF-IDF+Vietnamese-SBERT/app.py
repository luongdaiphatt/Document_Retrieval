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
import os

app = Flask(__name__)

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

# 1. Tiền xử lý văn bản
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

# 2. TF_IDF và SBERT tìm kiếm
def TF_IDF_search(query, k=50):
    vectorizer = TfidfVectorizer()
    corpus_tfidf = vectorizer.fit_transform(tokenized_corpus + [query])
    cosine_similarities = cosine_similarity(corpus_tfidf[-1], corpus_tfidf[:-1])[0]
    top_indices = cosine_similarities.argsort()[::-1][:k]
    top_similarities = cosine_similarities[top_indices]
    return top_indices.tolist(), top_similarities.tolist()

def SBERT_search(top_indices_tf_idf, query, k=50):
    # Get the top k indices from the TF-IDF search
    corpus = [tokenized_corpus[top_indices_tf_idf[i]] for i in range(k)]
    encoded_corpus = sbert.encode(corpus)
    encoded_query = sbert.encode(query)
    cossimilarity = []
    for i in range(encoded_corpus.shape[0]):
        cossimilarity.append((i, encoded_corpus[i, :].dot(encoded_query) / (np.linalg.norm(encoded_corpus[i, :]) * np.linalg.norm(encoded_query))))
    sorted_ids_data = sorted(cossimilarity, key=lambda x: x[1], reverse=True)
    top_indices = [top_indices_tf_idf[i[0]] for i in sorted_ids_data]
    top_similarities = [float(i[1]) for i in sorted_ids_data]
    return top_indices, top_similarities

# 3. BM25 và PhoBERT tìm kiếm
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

# 4. NDCG
def ndcg_at_k(scores, k):
    dcg = sum((2**scores[i] - 1) / np.log2(i + 2) for i in range(k))
    ideal_scores = sorted(scores, reverse=True)
    idcg = sum((2**ideal_scores[i] - 1) / np.log2(i + 2) for i in range(k))
    return dcg / idcg if idcg > 0 else 0

@app.route("/submit_scores", methods=["POST"])
def submit_scores():
    scores = list(map(int, request.form.getlist('scores')))
    k = len(scores)
    ndcg_score = ndcg_at_k(scores, k)
    
    # Store the scores and nDCG value
    with open('nDCG/scores.json', 'a') as f:
        f.write(json.dumps({'scores': scores, 'ndcg': ndcg_score, 'query': query, 'titles': [data[top_indices_tf_idf[i]]['title'] for i in items[:10]]}, ensure_ascii=False) + '\n')
    
    return json.dumps({'ndcg': ndcg_score})
    
@app.route("/load_scores", methods=["GET"])
def load_scores():
    try:
        with open('nDCG/scores.json', 'r') as f:
            data = json.load(f)
        return json.dump(data)
    except FileNotFoundError:
        return json.dump({'error': 'No scores found'})

# 5. Mô phỏng tìm kiếm
@app.route("/")
def home():
    return render_template("news/home.html")

@app.route("/search", methods=["POST", "GET"])
def submit():
    global query, top_indices_tf_idf, tf_idf_scores, top_indices_sbert, sbert_scores, top_indices_bm25, bm25_scores, top_indices_phobert, phobert_scores
    if request.method == "POST":
        query = request.form['search']
        p_query = preprocess_text(query)

        ### TF-IDF
        top_indices_tf_idf, tf_idf_scores = TF_IDF_search(p_query, 50)
        ### TF-IDF + SBERT
        top_indices_sbert, sbert_scores = SBERT_search(top_indices_tf_idf, p_query)

        ### BM25
        top_indices_bm25, bm25_scores = bm25_search(p_query, 50)
        ### BM25 + PhoBERT
        top_indices_phobert, phobert_scores = phobert_search(top_indices_bm25, p_query)

    # Chọn phương pháp tìm kiếm để hiển thị kết quả:
    # * TF-IDF
    # * TF-IDF + SBERT
    # * BM25
    # * BM25 + PhoBERT
    # Bỏ comment để chọn phương pháp tìm kiếm (1 trong 4)

    # Các query tiêu biểu để đánh giá: 
    # "cách kiếm tiền hiệu quả" 
    # "nước nào bổ"
    # "tác hại của chuối"

    ### TF-IDF: ưu tiên những từ khóa đơn lẻ có số lần xuất hiện nhiều + thứ tự các từ query không ảnh hưởng
    # page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')
    # pagination = Pagination(page=page, per_page=per_page, total=len(top_indices_tf_idf), css_framework='bootstrap5')
    # k_idx_show = top_indices_tf_idf[offset: offset + NUM_NEWS_PER_PAGE]
    # return render_template("news/search.html", k_idx=k_idx_show, data=data, similarities=tf_idf_scores, query=query, pagination=pagination)

    ### TF-IDF + SBERT: ưu tiên những từ khóa quan trọng và từ liên quan (nước <-> đồ uống, bổ <-> tốt, ...) + thứ tự query ảnh hưởng
    # page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')
    # pagination = Pagination(page=page, per_page=per_page, total=len(top_indices_sbert), css_framework='bootstrap5')
    # k_idx_show = top_indices_sbert[offset: offset + NUM_NEWS_PER_PAGE]
    # return render_template("news/search.html", k_idx=k_idx_show, data=data, similarities=sbert_scores, query=query, pagination=pagination)

    ### BM25: những từ khóa quan trọng sẽ được tìm kiếm chính xác hơn và dựa trên phần nhiều về độ liên quan + thứ tự query không ảnh hưởng
    # page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')
    # pagination = Pagination(page=page, per_page=per_page, total=len(top_indices_bm25), css_framework='bootstrap5')
    # k_idx_show = top_indices_bm25[offset: offset + NUM_NEWS_PER_PAGE]
    # return render_template("news/search.html", k_idx=k_idx_show, data=data, similarities=bm25_scores, query=query, pagination=pagination)

    ### BM25 + PhoBERT: tương tự như TF-IDF + SBERT nhưng sử dụng PhoBERT thay vì SBERT (thỉnh thoảng cho kết quả không liên quan như khi search "cá lóc")
    page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')
    pagination = Pagination(page=page, per_page=per_page, total=len(top_indices_phobert), css_framework='bootstrap5')
    k_idx_show = top_indices_phobert[offset: offset + NUM_NEWS_PER_PAGE]
    return render_template("news/search.html", k_idx=k_idx_show, data=data, similarities=phobert_scores, query=query, pagination=pagination)

# 6. API dùng để so sánh các phương pháp tìm kiếm và đánh giá
@app.route("/api/search", methods=["POST"])
def api_search():
    global items, query, top_indices_tf_idf
    if request.method == "POST":
        query = request.json['search']
        if 'k' in request.json:
            k = request.json['k']
        else:
            k = 20
        # print(query)
        p_query = preprocess_text(query)

        top_indices_tf_idf, tf_idf_scores = TF_IDF_search(p_query,k)
        top_indices_sbert, sbert_scores = SBERT_search(top_indices_tf_idf, p_query,k)

        top_indices_bm25, bm25_scores = bm25_search(p_query,k)
        top_indices_phobert, phobert_scores = phobert_search(top_indices_bm25, p_query,k)

        results = {
            "tf-idf": {
                "indices": top_indices_tf_idf,
                "scores": tf_idf_scores
            },
            "tf-idf+sbert": {
                "indices": top_indices_sbert,
                "scores": sbert_scores
            },
            "bm25": {
                "indices": top_indices_bm25,
                "scores": bm25_scores
            },
            "bm25+phobert": {
                "indices": top_indices_phobert,
                "scores": phobert_scores
            }
        }

        return json.dumps(results, ensure_ascii=False)

if __name__ == "__main__":

    rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"])
    stopwords = open(r"data/vietnamese-stopwords-dash.txt",'r',encoding='utf-8').read().split("\n")
    data = json.load(open(r'data/ArticlesNewspaper.json', 'r', encoding="utf-8"))

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download SBERT (save model if not exists)
    if not os.path.exists('sbert_model'):
        sbert = SentenceTransformer('keepitreal/vietnamese-sbert')
        sbert.save('sbert_model')
        print("SBERT downloaded")
    else:
        sbert = SentenceTransformer('sbert_model')
        print("SBERT loaded")

    # Download PhoBERT (save model if not exists)
    if not os.path.exists('phobert_model'):
        phobert = AutoModel.from_pretrained("vinai/phobert-base-v2", add_pooling_layer=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        phobert.save_pretrained('phobert_model')
        tokenizer.save_pretrained('phobert_model')
        print("PhoBERT downloaded")
    else:
        phobert = AutoModel.from_pretrained('phobert_model').to(device)
        tokenizer = AutoTokenizer.from_pretrained('phobert_model')
        print("PhoBERT loaded")
    
    # Save preprocessed data if not exists
    if not os.path.exists('data/tokenized_corpus.txt'):
        data_text = [i['title'] + " " + i['abstract'] for i in data]
        tokenized_corpus = [preprocess_text(corpus) for corpus in data_text]
        with open('data/tokenized_corpus.txt', 'w', encoding='utf-8') as f:
            for item in tokenized_corpus:
                f.write("%s\n" % item)
        print("Saved data")
    else:
        tokenized_corpus = open(r'data/tokenized_corpus.txt', 'r', encoding='utf-8').read().split("\n")
        print("Data loaded")

    # for BM25 search
    # BM25 Training (save trained model if not exists)
    if not os.path.exists('data/bm25_model.pkl'):
        split_tokenized_corpus = [doc.split() for doc in tokenized_corpus]
        bm25 = BM25Okapi(split_tokenized_corpus)
        # Save the BM25 model
        with open('data/bm25_model.pkl', 'wb') as bm25_file:
            pickle.dump(bm25, bm25_file)
        print("BM25 saved")
    else:
        with open('data/bm25_model.pkl', 'rb') as bm25_file:
            bm25 = pickle.load(bm25_file)
        print("BM25 loaded")

    # for PhoBERT search
    # Save the encoded corpus if not exists
    if not os.path.exists('data/encoded_corpus_phobert.npy'):
        encoded_corpus_phobert = encode_phobert(tokenized_corpus)
        np.save('data/encoded_corpus_phobert.npy', encoded_corpus_phobert)
        print("PhoBERT encoded corpus saved")
    else:
        encoded_corpus_phobert = np.load('data/encoded_corpus_phobert.npy')
        print("PhoBERT encoded corpus loaded")

    app.run(debug=True)