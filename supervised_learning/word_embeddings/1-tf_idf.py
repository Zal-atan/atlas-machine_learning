#!/usr/bin/env python3
""" Module which creates td_idf function """
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF Embedding

    Inputs:
    sentences: list of sentences to analyze
    vocab: list of the vocabulary words to use for the analysis
        If None, all words within sentences should be used

    Returns:
    embeddings: numpy.ndarray of shape (s, f) containing the embeddings
        s: number of sentences in sentences
        f: number of features analyzed
    features: list of the features used for embeddings
    """

    if vocab is None:
        vectorize = TfidfVectorizer()
        sen = vectorize.fit_transform(sentences)
        vocab = vectorize.get_feature_names_out()

    else:
        vectorize = TfidfVectorizer(vocabulary=vocab)
        sen = vectorize.fit_transform(sentences)

    embedding = sen.toarray()

    return embedding, vocab
