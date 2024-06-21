#!/usr/bin/env python3
""" Module for creating variational autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder

    Inputs:
    input_dims: integer containing the dimensions of the model input
    hidden_layers: list containing the number of nodes for each hidden layer
    in the encoder, respectively\\
        the hidden layers should be reversed for the decoder
    latent_dims: integer containing the dimensions of the latent space
    representation

    Returns: encoder, decoder, auto
    encoder: encoder model
    decoder: decoder model
    auto: full autoencoder model
    """

    import pdb
    pdb.set_trace()

    # Build encoder
    input_image = keras.Input(shape=(input_dims,))
    encode = input_image

    for layer in hidden_layers:
        encode = keras.layers.Dense(layer, activation='relu')(encode)

        # Latent Space
    mean_latent = keras.layers.Dense(latent_dims, activation=None)(encode)
    log_vari_latent = keras.layers.Dense(
        latent_dims, activation=None)(encode)

    def sampling(args):
        """Sampling points in latent space"""
        mean_latent, log_vari_latent = args
        batch = keras.backend.shape(mean_latent)[0]
        dim = keras.backend.shape(mean_latent)[1]
        ep = keras.backend.random_normal(shape=(batch, dim))

        return mean_latent + keras.backend.exp(log_vari_latent / 2) * ep

    latent_space = keras.layers.Lambda(
        sampling)([mean_latent, log_vari_latent])

    encoder = keras.Model(input_image, [latent_space,
                                        mean_latent,
                                        log_vari_latent])

    # Build Decoder
    decode_image_size = keras.Input(shape=(latent_dims,))
    decode = decode_image_size

    for layer in reversed(hidden_layers):
        decode = keras.layers.Dense(layer, activation='relu')(decode)
    decode = keras.layers.Dense(input_dims, activation='sigmoid')(decode)
    decoder = keras.Model(decode_image_size, decode, name='decoder')

    # AutoEncoder
    encode_output = encoder(input_image)[0]
    decode_output = decoder(encode_output)
    autoencoder = keras.Model(
        inputs=input_image,
        outputs=decode_output,
        name='autoencoder')

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
