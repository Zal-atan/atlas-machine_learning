#!/usr/bin/env python3
"""Create the gensim_to keras function"""
from tensorflow.keras.layers import Embedding


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a keras Embedding layer

    Inputs:
    model: trained gensim word2vc model

    Return:
    trainable keras embedding
    """

    # Extract the keyed vectors (word vectors) from the gensim model
    keyed_vectors = model.wv

    # Retrieve the weight matrix (vectors for each word)
    embed_weights = keyed_vectors.vectors

    # Create a Keras embed layer with the weights from the gensim model
    embed_layer = Embedding(
        input_dim=embed_weights.shape[0],  # Number of words (vocabulary size)
        output_dim=embed_weights.shape[1],
        # Dimensionality of the word vectors
        weights=[embed_weights],           # Initialize with gensim weights
        trainable=False,                       # Set to non-trainable
    )

    return embed_layer
