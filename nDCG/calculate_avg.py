import json
from statistics import mean
import pandas as pd

datas = json.loads(open('scores.json').read())

tf_idf_scores = []
tf_idf_sbert_scores = []
bm25_scores = []
bm25_phobert_scores = []
for data in datas:
  tf_idf_scores.append(data['tf-idf'])
  tf_idf_sbert_scores.append(data['tf-idf+sbert'])
  bm25_scores.append(data["bm25"])
  bm25_phobert_scores.append(data["bm25+phobert"])

print(mean(tf_idf_scores))
print(mean(tf_idf_sbert_scores))
print(mean(bm25_scores))
print(mean(bm25_phobert_scores))

data_excel = {
  "Model" : ["tf-idf","tf-idf+sbert","bm25","bm25+phobert"],
  "tf-idf": [mean(tf_idf_scores), mean(tf_idf_sbert_scores), mean(bm25_scores), mean(bm25_phobert_scores)]
}

df = pd.DataFrame(data_excel)
df.to_excel("average_scores.xlsx", index=False)
