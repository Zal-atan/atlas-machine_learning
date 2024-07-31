#!/usr/bin/env python3
""" Module for creating the cumulative_bleu function"""
import numpy as np
from scipy.stats import gmean
ngram_bleu = __import__('1-ngram_bleu').ngram_bleu


def cumulative_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence

    Inputs:\\
    references: list of reference translations\\
        Each reference translation is a list of the words in the translation\\
    sentence: list containing the model proposed sentence\\
    n: size of the n-gram to use for evaluation\\

    Returns:\\
    the cumulative n-gram BLEU score
    """
    n_gram_scores = []

    # Calculate the BLEU score for each n-gram from 1 to n
    for i in range(1, n + 1):
        n_gram_scores.append(ngram_bleu(references, sentence, i))

    return gmean(n_gram_scores)
