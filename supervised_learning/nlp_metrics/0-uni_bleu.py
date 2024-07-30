#!/usr/bin/env python3
""" Module for creating the uni_bleu function"""
import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the n-gram BLEU score for a sentence

    Inputs:\\
    references: list of reference translations\\
        Each reference translation is a list of the words in the translation\\
    sentence: list containing the model proposed sentence\\

    Returns:\\
    the unigram BLEU score
    """
    # Initialize the length of the proposed sentence
    proposed_length = len(sentence)

    match_count = 0

    matched_unigrams = []

    # Check for the matching words
    for reference in references:
        for word in sentence:
            if word in reference and word not in matched_unigrams:
                matched_unigrams.append(word)

    # Count the number of matched unigrams
    match_count = len(matched_unigrams)

    precision = match_count / proposed_length

    # Calculate the length of the shortest reference translation
    shortest_reference_length = min(len(reference) for reference in references)

    # Calculate the brevity penalty (BP)
    if proposed_length < shortest_reference_length:
        # if sentence shorter then reference, apply brevity penalty
        brevity_penalty = np.exp(
            1 - (shortest_reference_length / proposed_length))
    else:
        brevity_penalty = 1

    return precision * brevity_penalty
