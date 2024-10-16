from flask import Flask, request, jsonify, render_template
from search_query import DocumentRetrieval

app = Flask(__name__)

# Simulated search function

# initiate the search query engine
dr = DocumentRetrieval()
dr.init('data/crawled.csv', 'data/vietnamese-stopwords.txt')

# Simulated search function with more detailed results
def perform_search(query, page_number=1):
    results, similarities = dr.search(query, page_number=page_number)
    return results

# Route to render the HTML page


@app.route('/')
def index():
    return render_template('index.html')


# Route to handle search and return results


@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query', '')
    if query:
        results = perform_search(query)
        return jsonify(results=results)
    return jsonify(results=[])


if __name__ == '__main__':
    app.run(debug=True)
