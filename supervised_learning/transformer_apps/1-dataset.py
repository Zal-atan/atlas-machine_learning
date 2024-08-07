#!/usr/bin/env python3
""" Module creating a Dataset Class """

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    """Loads and preps a dataset for machine translation"""

    def __init__(self):
        """ Initialize Dataset Class"""

        examples, _ = tfds.load('ted_hrlr_translate/pt_to_en',
                                with_info=True,
                                as_supervised=True)

        self.data_train, self.data_valid = examples['train'], \
            examples['validation']
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset. Maximum Vocab size should
        be set to 2**15

        Input:
        data: tf.data.Dataset whose examples are formatted as a tuple
        (pt, en)\\
            pt: tf.Tensor containing the Portuguese sentence\\
            en: tf.Tensor containing the corresponding English sentence

        Returns:
            tokenizer_pt: The Portuguese tokenizer
            tokenizer_en: The English tokenizer
        """
        tokenizer_pt = tfds.deprecated.text.\
            SubwordTextEncoder.build_from_corpus(
                (pt.numpy() for pt, _ in data), target_vocab_size=2**15)
        tokenizer_en = tfds.deprecated.text.\
            SubwordTextEncoder.build_from_corpus(
                (en.numpy() for _, en in data), target_vocab_size=2**15)

        return tokenizer_pt, tokenizer_en
    
    def encode(self, pt, en):
        """
        Encodes a translation into tokens

        Inputs:\\
        pt: tf.Tensor containing the Portuguese sentence\\
        en: tf.Tensor containing the corresponding English sentence

        Return:\\
        pt_tokens: np.ndarray containing the Portuguese tokens\\
        en_tokens: np.ndarray. containing the English tokens
        """
        port_start = [self.tokenizer_pt.vocab_size]
        port_end = [self.tokenizer_pt.vocab_size + 1]
        eng_start = [self.tokenizer_en.vocab_size]
        eng_end = [self.tokenizer_en.vocab_size + 1]

        port_tokens = port_start + self.tokenizer_pt.encode(pt.numpy()) +\
            port_end
        eng_tokens = eng_start + self.tokenizer_en.encode(en.numpy()) + \
            eng_end

        return port_tokens, eng_tokens
