#!/usr/bin/env python3
""" Module for creating sparse autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates a sparse autoencoder

    Inputs:
    input_dims: integer containing the dimensions of the model input
    hidden_layers: list containing the number of nodes for each hidden layer in
    the encoder, respectively the hidden layers should be reversed
    for the decoder
    latent_dims: integer containing the dimensions of the latent space
    representation
    lambtha: regularization parameter used for L1 regularization
    on the encoded output

    Returns: encoder, decoder, auto
    encoder: encoder model
    decoder: decoder model
    auto: full autoencoder model
    """

    regularize = keras.regularizers.l1(lambtha)
    input_image = keras.Input(shape=(input_dims,))

    # Build encoder
    encode = input_image
    for layer in hidden_layers:
        encode = keras.layers.Dense(layer, activation='relu')(encode)
    encode = keras.layers.Dense(latent_dims, activation='relu',
                                activity_regularizer=regularize)(encode)
    encoder = keras.Model(input_image, encode, name='encoder')

    # Build Decoder
    decode_image_size = keras.Input(shape=(latent_dims,))
    decode = decode_image_size
    for layer in hidden_layers[::-1]:
        decode = keras.layers.Dense(layer, activation='relu')(decode)
    decode = keras.layers.Dense(input_dims, activation='sigmoid')(decode)
    decoder = keras.Model(decode_image_size, decode, name='decoder')

    # AutoEncoder
    encode_output = encoder(input_image)
    decode_output = decoder(encode_output)
    autoencoder = keras.Model(
        inputs=input_image,
        outputs=decode_output,
        name='autoencoder')

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
