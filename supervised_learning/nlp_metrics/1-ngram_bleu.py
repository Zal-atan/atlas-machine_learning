#!/usr/bin/env python3
""" Module for creating the ngram_bleu function"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence

    Inputs:\\
    references: list of reference translations\\
        Each reference translation is a list of the words in the translation\\
    sentence: list containing the model proposed sentence\\
    n: size of the n-gram to use for evaluation\\

    Returns:\\
    the n-gram BLEU score
    """
    # Calculate n-gram counts in the sentence
    count_dict = {}
    for i in range(len(sentence) - n + 1):
        ngram = tuple(sentence[i:i + n])
        count_dict[ngram] = count_dict.get(ngram, 0) + 1
    # proposed_length is the total number of unique n-grams in the sentence
    proposed_length = len(count_dict)

    # Calculate maximum n-gram counts in the references
    max_counts = {}
    for reference in references:
        match_count = {}
        for i in range(len(reference) - n + 1):
            ngram = tuple(reference[i:i + n])
            match_count[ngram] = match_count.get(ngram, 0) + 1
        for ngram, count in match_count.items():
            max_counts[ngram] = max(max_counts.get(ngram, 0), count)

    # Calculate clipped n-gram counts
    clipped_counts = {}
    for ngram, count in count_dict.items():
        clipped_counts[ngram] = min(count, max_counts.get(ngram, 0))
    # m is the total number of clipped n-grams
    clipped = sum(clipped_counts.values())

    # Calculate precision as the ratio of clipped n-gram counts to total
    # n-gram counts in the sentence
    P = clipped / proposed_length

    ref_len = min(len(reference) for reference in references)
    c = len(sentence)
    # Calculate BP; if sentence is shorter than the shortest reference, apply
    # penalty
    BP = min(1, np.exp(1 - ref_len / c))

    return P * BP
