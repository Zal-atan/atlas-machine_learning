#!/usr/bin/env python3
""" This module creates the Q/a Chatbot """

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer a question

    Inputs:\\
    question: string containing the question to answer\\
    reference: string containing the reference document from which to
    find the answer

    Returns:
    A string containing the answer
    """
    # Load pre-trained BERT and tokenizer
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')

    # Tokenize question text and reference text
    que_tokens = tokenizer.tokenize(question)
    ref_tokens = tokenizer.tokenize(reference)

    # Combine tokens adding special tokens, then convert into ids
    tokens = ['[CLS]'] + que_tokens + ['[SEP]'] + ref_tokens + ['[SEP]']
    word_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Create mask for input, and segment the ID's to distinguish q's from ref's
    mask_input = [1] * len(word_ids)
    id_types = [0] * (1 + len(que_tokens) + 1) + [1] * (len(ref_tokens) + 1)

    # Convert the list to tensors
    word_ids, mask_input, id_types = map(lambda t: tf.expand_dims(
        tf.convert_to_tensor(t, dtype=tf.int32), 0),
        (word_ids, mask_input, id_types))

    # Pass into BERT
    outputs = model([word_ids, mask_input, id_types])

    # Find positions of start and end in answer text
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1

    # Extract the tokens, then convert to string for answer
    ans_tokens = tokens[short_start: short_end + 1]

    return tokenizer.convert_tokens_to_string(ans_tokens)
