#!/usr/bin/env python3
"""Module which trains a convolutional neural network to classify the 
CIFAR 10 dataset"""
import tensorflow.keras as K


epochs = 15
batch_size = 64
img_reshape = 160

def preprocess_data(X, Y):
    """
    Pre-processes the data for your model

    Inputs:
    X - numpy.ndarray (m, 32, 32, 3) containing the CIFAR 10 data, where
        m is the number of data points
    Y - numpy.ndarray (m,) containing the CIFAR 10 labels for X

    Returns:
    X_p - numpy.ndarray containing the preprocessed X
    Y_p - numpy.ndarray containing the preprocessed Y
    """
    X_p = K.applications.densenet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    inputs = K.Input(shape=(32, 32, 3))
    input_resized = K.layers.Lambda(lambda image: K.backend.resize_images(
        image, img_reshape/32, img_reshape/32, "channels_last"))(inputs)
    
    base_model = K.applications.DenseNet121(include_top=False,
                                             weights='imagenet',
                                             input_tensor=input_resized,
                                             input_shape=(img_reshape,
                                                          img_reshape, 3),
                                             pooling='max')
    base_model.trainable = False

    model = K.models.Sequential()
    model.add(base_model)
    model.add(K.layers.BatchNormalization())
    # model.add(K.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(K.layers.Dropout(0.25))


    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(512, activation=('relu')))
    # model.add(K.layers.Dropout(0.5))
    # model.add(K.layers.Dense(128, activation=('relu')))
    model.add(K.layers.Dropout(0.2))
    model.add(K.layers.Dense(10, activation=('softmax')))
    callback = []

    learn_rate_reduce = K.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                        factor=.11,
                                        patience=2,
                                        min_lr=1e-5)
    callback.append(learn_rate_reduce)


    callback.append(K.callbacks.ModelCheckpoint('cifar10.h5',
                                                monitor='val_accuracy',
                                                save_best_only=True,
                                                mode='max'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test, Y_test), callbacks=callback)

    model.save('cifar10.h5')

