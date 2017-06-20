import numpy as np
from scipy.spatial.distance import pdist, cdist
from gensim import utils
from gensim.models.word2vec import Word2Vec
from gensim.models.tfidfmodel import TfidfModel

DEFAULT_SAMPLE_SIZE = 50

def sample(w, size, pwr = 1.5):
    t,f = w[:,0], w[:,1]**pwr
    p = f/np.linalg.norm(f, 1)
    return np.random.choice(t, size = size, replace = True, p = p)

def get_wv(model, w):
    try:
        return model[w]
    except KeyError:
        return None

def doc_vec(doc, model, corpus, size = DEFAULT_SAMPLE_SIZE, tfidf = None, count = 1):
    """ Creates a document vector """

    tfidf = tfidf or TfidfModel(corpus)

    # w is Dx2 array with word id and tfidf score
    w = np.array(tfidf[corpus.dictionary.doc2bow(doc)])

    # sample according to tfidf scores and get vectors,
    # filter all not-found words
    vecs = [get_wv(model, corpus.dictionary[x]) for x in sample(w, size)]
    vecs = [v for v in vecs if v is not None]

    # Handling the cases when we find very few words from a document
    # in our externally trained model vocabulary
    if len(vecs) < .5*size:
        if count < 5:
            return doc_vec(doc, model, corpus, size, tfidf, count + 1)
        else:
            raise KeyError("Cannot find any of these words in the vocabulary: " + " ".join(doc))

    # Just take the mean of the vec of all the sampled words from the document
    return np.mean(vecs, 0)

def corpus_vec(docs, model, corpus, size = DEFAULT_SAMPLE_SIZE):
    """ Creates a NxD array of document vectors for each document in a list"""

    tfidf = TfidfModel(corpus)
    N,D = len(docs), model.wv.syn0.shape[1]
    arr = np.empty((N, D))
    for i in range(N):
        arr[i,:] = doc_vec(docs[i], model, corpus, size, tfidf)
    return arr

def get_closest_doc(v, cv, docs):
    """ given a vector and 2D array of corpus vectors gives best cv"""
    v = np.array([v])
    d = np.argsort(cdist(v, cv)[0,:])
    i = d[0]
    return i, ' '.join(docs[i])

def sentence_to_vec(text, model, corpus, tfidf = None):
    doc = list(utils.tokenize(text))
    return doc_vec(doc, model, corpus, tfidf = tfidf)
