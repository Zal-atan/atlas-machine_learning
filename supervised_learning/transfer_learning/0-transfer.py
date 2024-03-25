#!/usr/bin/env python3
"""Module which trains a convolutional neural network to classify the 
CIFAR 10 dataset"""
import tensorflow.keras as K


epochs = 25
batch_size = 128
# img_reshape = 160
# input_shape = (img_reshape, img_reshape, 3)

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
        image, 160/32, 160/32, "channels_last"))(inputs)
    
    # base_model = K.applications.DenseNet121(include_top=False,
    #                                          weights='imagenet',
    #                                          input_tensor=input_resized,
    #                                          input_shape=(160, 160, 3),
    #                                          pooling='avg')
    # base_model.trainable = False
    base_model = K.applications.InceptionV3(include_top=False,
                                            weights='imagenet',
                                            input_tensor=input_resized,
                                            input_shape=(160, 160, 3),
                                            pooling='avg')

    model = K.models.Sequential()
    model.add(base_model)

    model.add(K.layers.Flatten())

    # model.add(K.layers.Dense(256, activation=('relu')))
    model.add(K.layers.Dense(256))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Activation('relu'))
    model.add(K.layers.Dropout(0.5))

    # model.add(K.layers.Dense(128, activation=('relu')))
    model.add(K.layers.Dense(128))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Activation('relu'))
    model.add(K.layers.Dropout(0.35))

    # model.add(K.layers.Dense(64, activation=('relu')))
    model.add(K.layers.Dense(64))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Activation('relu'))
    model.add(K.layers.Dropout(0.2))

    model.add(K.layers.Dense(10, activation=('softmax')))
    callback = []

    learn_rate_reduce = K.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                        factor=.01,
                                        patience=2,
                                        min_lr=1e-5)
    callback.append(learn_rate_reduce)

    model_checkpoint = K.callbacks.ModelCheckpoint('cifar10.h5',
                                                monitor='val_accuracy',
                                                save_best_only=True,
                                                mode='max')
    callback.append(model_checkpoint)

    early_stopping = K.callbacks.EarlyStopping(monitor='val_accuracy',
                                               mode='max',
                                               patience=5)
    callback.append(early_stopping)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test, Y_test), callbacks=callback)

    model.save('cifar10.h5')

