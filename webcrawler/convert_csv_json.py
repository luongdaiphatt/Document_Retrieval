import csv
import json

csv_read = csv.DictReader(open('crawled.csv', 'r', encoding='utf-8-sig'))
json_file = open('crawled.json', 'w')
json_write = json.dumps([row for row in csv_read], ensure_ascii=False, indent=4)
json_file.write(json_write)