#!/usr/bin/env python3
""" Module for creating convolutional autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder

    Inputs:
    input_dims: integer containing the dimensions of the model input
    filters: list containing the number of filters for each convolutional layer
    in the encoder, respectively\
    the filters should be reversed for the decoder
    latent_dims: integer containing the dimensions of the latent space
    representation

    Returns: encoder, decoder, auto
    encoder: encoder model
    decoder: decoder model
    auto: full autoencoder model
    """

    input_image = keras.Input(shape=input_dims)

    # Build encoder
    encode = input_image
    for layer in range(len(filters)):
        encode = keras.layers.Conv2D(filters[layer], (3, 3), padding='same',
                                     activation='relu')(encode)
        encode = keras.layers.MaxPooling2D((2, 2), padding='same')(encode)
    encoder = keras.Model(input_image, encode, name='encoder')

    # Build Decoder
    decode_image_size = keras.Input(shape=latent_dims)
    decode_layers = filters[::-1]
    decode = decode_image_size
    for layer in range(len(filters)):
        if layer != len(filters) - 1:
            decode = keras.layers.Conv2D(decode_layers[layer], (3, 3),
                                         padding='same',
                                         activation='relu')(decode)
            decode = keras.layers.UpSampling2D((2, 2))(decode)
        else:
            decode = keras.layers.Conv2D(decode_layers[layer],
                                         (3, 3), padding='valid',
                                         activation='relu')(decode)
            decode = keras.layers.UpSampling2D((2, 2))(decode)
    decode = keras.layers.Conv2D(input_dims[2], (3, 3), activation='sigmoid',
                                 padding='same')(decode)
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
