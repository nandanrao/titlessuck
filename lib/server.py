from __future__ import print_function
from flask import Flask, request
from flask_cors import CORS, cross_origin
from dotenv import load_dotenv
import json
from gensim.corpora import TextCorpus
from gensim.models.word2vec import Word2Vec
from gensim.models.tfidfmodel import TfidfModel
from jobs import sentence_to_vec, get_closest_doc, corpus_vec

# env vars
load_dotenv('./.env')

# logging
import logging
logger = logging.getLogger()
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

logger.info("Setting up app!")

app = Flask(__name__)
CORS(app)

# setup model stuff
corpus = TextCorpus('jobspicker/jobspicker-descriptions.csv')
corpus.dictionary.filter_extremes(no_below=4, no_above=.9, keep_n=100000)
sentences = [list(g) for g in list(corpus.get_texts())]
tfidf = TfidfModel(corpus)
model = Word2Vec.load("profiles.model")
corp_vecs = corpus_vec(sentences, model, corpus)

# create simple helper functions
get_vec = lambda t: sentence_to_vec(t, model, corpus, tfidf)
get_job = lambda v: get_closest_doc(v, corp_vecs, sentences)

# our database of bayesopt models
user_models = {}

@app.route('/init/<i>')
def init(i):
    # make bayesianopt class with id and store in memory user_modes[i] =
    user_models[i] = "foo"
    return "Setup with id: " + i

@app.route('/describe/<i>', methods = ['POST'])
def describe(i):
    data = request.json
    txt = data['description']

    vec = get_vec(txt)
    job_id, job = get_job(vec)
    return json.dumps({ 'jobId': job_id, 'description': job })

@app.route('/rate/<i>', methods = ['POST'])
def rate(i):
    data = request.json
    rating = int(data['rating'])
    job_id = int(data['jobId'])
    model = user_models[i]
    # update bayesopt model with job_id/rating
    # get new suggestion from bayesopt model
    vec = []
    job_id, job = get_job(vec)
    return json.dumps({ 'jobId': job_id, 'description': job })

def run():
    port = os.environ.get('FLASK_PORT') or 5000
    app.run(port = port)
    logger.info("App listening on port: " + port)
