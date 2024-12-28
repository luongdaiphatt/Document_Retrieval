import json


scores = []
for line in open('scores.json', 'r').readlines():
    data = json.loads(line.strip())
    scores.append(float(data['ndcg']))

print('Average nDCG:', sum(scores) / len(scores))