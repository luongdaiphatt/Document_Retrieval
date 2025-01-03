import requests
import os
import json
from flask import Flask, render_template, request
import numpy as np

# RUN THIS AFTER RUNNING THE SEARCH ENGINE (app.py)
# This script is used to calculate nDCG@k for the search engine at http://localhost:5001

app = Flask(__name__)

def search_query(query, k=20):
  url = 'http://localhost:5000/api/search'
  payload = {'search': query, 'k': k}
  response = requests.post(url, json=payload)
  
  if response.status_code == 200:
    return response.json()
  else:
    return {'error': 'Failed to retrieve data'}

# Function to get titles by merged indices
def get_titles_by_indices(indices):
  titles = []
  for idx in indices:
    titles.append(data[idx]['title'])
  return titles

def get_abstracts_by_indices(indices):
  abstracts = []
  for idx in indices:
    abstracts.append(data[idx]['abstract'])
  return abstracts

def ndcg_at_k(scores, k):
    dcg = sum((2**scores[i] - 1) / np.log2(i + 2) for i in range(k))
    ideal_scores = sorted(scores, reverse=True)
    idcg = sum((2**ideal_scores[i] - 1) / np.log2(i + 2) for i in range(k))
    return dcg / idcg if idcg > 0 else 0

def merge_indices(indices1, indices2):
  return list(set(indices1 + indices2))

@app.route('/')
def index():
  return render_template('ndcg/index.html')

@app.route('/search', methods=['POST'])
def search():
  global tf_idf_in_merged, tf_idf_sbert_in_merged, bm25_in_merged, bm25_phobert_in_merged, merged_indices, query, titles, abstracts
  query = request.form['query']
  search_results = search_query(query)
  # Merge indices from TF-IDF and BM25 without duplicates
  merged_indices = merge_indices(search_results['tf-idf']['indices'], search_results['bm25']['indices'])
  tf_idf_in_merged = [merged_indices.index(idx) for idx in search_results['tf-idf']['indices']]
  tf_idf_sbert_in_merged = [merged_indices.index(idx) for idx in search_results['tf-idf+sbert']['indices']]
  bm25_in_merged = [merged_indices.index(idx) for idx in search_results['bm25']['indices']]
  bm25_phobert_in_merged = [merged_indices.index(idx) for idx in search_results['bm25+phobert']['indices']]
  titles = get_titles_by_indices(merged_indices)
  abstracts = get_abstracts_by_indices(merged_indices)
  return render_template('ndcg/search.html', query=query, titles=titles, abstracts=abstracts)

@app.route('/search', methods=['GET'])
def search_get():
  return render_template('ndcg/index.html')

@app.route('/submit', methods=['POST'])
def submit():
  scores = list(map(int, request.form.getlist('scores')))
  # map the scores back to the indices
  tf_idf_scores = [scores[idx] for idx in tf_idf_in_merged]
  tf_idf_sbert_scores = [scores[idx] for idx in tf_idf_sbert_in_merged]
  bm25_scores = [scores[idx] for idx in bm25_in_merged]
  bm25_phobert_scores = [scores[idx] for idx in bm25_phobert_in_merged]
  ndcg_tf_idf = ndcg_at_k(tf_idf_scores, len(tf_idf_scores))
  ndcg_tf_idf_sbert = ndcg_at_k(tf_idf_sbert_scores, len(tf_idf_sbert_scores))
  ndcg_bm25 = ndcg_at_k(bm25_scores, len(bm25_scores))
  ndcg_bm25_phobert = ndcg_at_k(bm25_phobert_scores, len(bm25_phobert_scores))
  
  print('nDCG@k for TF-IDF:', ndcg_tf_idf)
  print('nDCG@k for TF-IDF+SBERT:', ndcg_tf_idf_sbert)
  print('nDCG@k for BM25:', ndcg_bm25)
  print('nDCG@k for BM25+PhoBERT:', ndcg_bm25_phobert)
  results = {
    'tf-idf': ndcg_tf_idf,
    'tf-idf+sbert': ndcg_tf_idf_sbert,
    'bm25': ndcg_bm25,
    'bm25+phobert': ndcg_bm25_phobert
  }
  
  # Load the old list from scores.json if it exists
  os.makedirs('nDCG', exist_ok=True)
  scores_file = 'nDCG/scores.json'
  if os.path.exists(scores_file):
    with open(scores_file, 'r', encoding='utf-8') as f:
      old_data = json.load(f)
  else:
    old_data = []

  # Append the new result to the old list
  new_result = {
    'query': query,
    'titles': get_titles_by_indices(merged_indices),
    'scores': scores,
    'tf-idf': ndcg_tf_idf,
    'tf-idf+sbert': ndcg_tf_idf_sbert,
    'bm25': ndcg_bm25,
    'bm25+phobert': ndcg_bm25_phobert
  }
  old_data.append(new_result)

  # Calculate average nDCG@k for each model
  avg_results = {
    'tf-idf': np.mean([result['tf-idf'] for result in old_data]),
    'tf-idf+sbert': np.mean([result['tf-idf+sbert'] for result in old_data]),
    'bm25': np.mean([result['bm25'] for result in old_data]),
    'bm25+phobert': np.mean([result['bm25+phobert'] for result in old_data])
  }

  # Save the updated list back to scores.json
  with open(scores_file, 'w', encoding='utf-8') as f:
    json.dump(old_data, f, ensure_ascii=False, indent=4)
  return render_template('ndcg/scores.html', query=query, titles=titles, abstracts=abstracts, scores=scores, results=results, avg_results=avg_results)

@app.route('/scores', methods=['GET'])
def scores():
  scores_file = 'nDCG/scores.json'
  if os.path.exists(scores_file):
    with open(scores_file, 'r', encoding='utf-8') as f:
      old_data = json.load(f)
  else:
    old_data = []
  return render_template('ndcg/scores.html', scores=old_data)

if __name__ == "__main__":
  data = json.load(open(r'data/ArticlesNewspaper.json', 'r', encoding="utf-8"))
  tokenized_corpus = open(r'data/tokenized_corpus.txt', 'r', encoding='utf-8').read().split("\n")
  print("Data loaded")
  
  app.run(port=5001)

