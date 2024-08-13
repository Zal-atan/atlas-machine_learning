#!/usr/bin/env python3
""" This module creates the semantic_search function """

import tensorflow as tf
import tensorflow_hub as hub
import os
import numpy as np


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents

    Inputs:\\
    corpus_path: path to the corpus of reference documents on which
        to perform semantic search\\
    sentence: sentence from which to perform semantic search

    Return:
    The reference text of the document most similar to sentence
    """

    # Load Universal Sentence Encoder Model
    model = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-large/5")

    # Load the corpus docs into a list which will become array
    files = []
    for file in os.listdir(corpus_path):
        if not file.endswith(".md"):
            continue
        with open(corpus_path + "/" + file, "r", encoding="utf-8") as f:
            files.append(f.read())

    # Compute embeddings for all the corpus documents and inital sentence
    corp_embed = model(files)
    sent_embed = model([sentence])[0]

    # Normalize embeddings to unit vectors
    corp_embed = tf.nn.l2_normalize(corp_embed, axis=1)
    sent_embed = tf.nn.l2_normalize(sent_embed, axis=0)

    # Compute cosine similarity between embedding of sentence and each corp doc
    correlate = np.inner(sent_embed, corp_embed)

    # Find the index of the doc in corp that has highest similarity
    close = np.argmax(correlate)

    # Retrieve most similar doc text
    similarity = files[close]

    return similarity
