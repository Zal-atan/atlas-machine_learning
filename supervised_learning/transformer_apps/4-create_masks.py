#!/usr/bin/env python3
""" Module creating the create_masks function"""

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


def create_masks(inputs, target):
    """
    Creates all masks for training/validation

    Inputs:\\
    inputs: tf.Tensor of shape (batch_size, seq_len_in) that
        contains the input sentence\\
    target: tf.Tensor of shape (batch_size, seq_len_out) that contains
        he target sentence\\

    Returns:\\
    encoder_mask: tf.Tensor padding mask of shape
        (batch_size, 1, 1, seq_len_in) to be applied in the encoder\\
    combined_mask: tf.Tensor of shape (batch_size, 1, seq_len_out, seq_len_out)
        used in the 1st attention block in the decoder to pad and mask future
        tokens in the input received by the decoder. It takes the maximum
        between a lookaheadmask and the decoder target padding mask.\\
    decoder_mask: tf.Tensor padding mask of shape
        (batch_size, 1, 1, seq_len_in) used in the 2nd attention block
        in the decoder.\\
    """
    # create encode mask to mask out padding and add extra dimensions
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    # look-ahead mask to mask out future tokens in target
    size = target.shape[1]
    look_ahead = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

    # create target mask to mask out padding and add extra dimensions
    target_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    target_mask = target_mask[:, tf.newaxis, tf.newaxis, :]

    # combine look-ahead and target-mask into combined
    combined_mask = tf.maximum(target_mask, look_ahead)

    # create decoder mask to mask out padding and add extra dimensions
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    return encoder_mask, combined_mask, decoder_mask
