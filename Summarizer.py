# summarizer.py

import argparse
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.probability import FreqDist
from heapq import nlargest
from collections import defaultdict

from flask import Flask, render_template, request

app = Flask(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--length', default=4, type=int, help='No. of sentences to return')
    parser.add_argument('-t', '--text_content', help='Text content to summarize')
    args = parser.parse_args()
    return args

def sanitize_input(data):
    replace = {
        ord('\f'): ' ',
        ord('\t'): ' ',
        ord('\n'): ' ',
        ord('\r'): None
    }
    return data.translate(replace)

def tokenize_content(content):
    stop_words = set(stopwords.words('english') + list(punctuation))
    words = word_tokenize(content.lower())
    return (sent_tokenize(content), [word for word in words if word not in stop_words])

def score_tokens(sent_tokens, word_tokens):
    word_freq = FreqDist(word_tokens)
    rank = defaultdict(int)
    for i, sent in enumerate(sent_tokens):
        for word in word_tokenize(sent.lower()):
            if word in word_freq:
                rank[i] += word_freq[word]
    return rank

def summarize(ranks, sentences, length):
    if length > len(sentences):
        length = len(sentences)  # Adjust length to match the number of sentences in the text
    indices = nlargest(length, ranks, key=ranks.get)
    final_summary = [sentences[j] for j in indices]
    return final_summary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize_text():
    text = request.form['text_content']
    line_count = int(request.form['line_count'])
    content = sanitize_input(text)
    sent_tokens, word_tokens = tokenize_content(content)
    sent_ranks = score_tokens(sent_tokens, word_tokens)
    summary = summarize(sent_ranks, sent_tokens, line_count)
    return render_template('summary.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
