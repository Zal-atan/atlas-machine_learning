#!/usr/bin/env python3
""" Module for predicting Bitcoin future hour data"""

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class GenerateWindow():
    """ Generates a 24 hour window for predicting"""

    def __init__(self, input_width, label_width, shift,
                 train_data, valid_data, test_data,
                 label_columns=None):

        # Storing the raw data.
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_data.columns)}

        # window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack([labels[:, :, self.column_indices[name]]
                               for name in self.label_columns], axis=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col='Close', max_subplots=4):
        """ Plots the expected vs the real"""
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))

        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(
                    plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)

            if model is not None:
                predictions = model(inputs)
                # Ensure predictions shape matches label width
                predictions = predictions[:, -self.label_width:, :]
                # Remove the last dimension if it's 1
                predictions = np.squeeze(predictions, axis=-1)
                # print(f"Shape of predictions: {predictions.shape}")
                # print(f"Shape of label indices: {self.label_indices.shape}")
                # print(f"Shape of labels: {labels.shape}")
                plt.scatter(self.label_indices, predictions[n, :],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')
        save_path = "./predictions_plot"
        plt.savefig(save_path, format='png')
        plt.show()

    def make_dataset(self, data):
        """ Turns DF into Keras dataset"""
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_data)

    @property
    def val(self):
        return self.make_dataset(self.valid_data)

    @property
    def test(self):
        return self.make_dataset(self.test_data)

    @property
    def example(self):
        """ Get and cache an example batch of `inputs, labels` for plotting. """
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
            return result


def create_model_and_compile(model, window, patience=2, epochs=20):
    """
    Creates, compiles, and fits the data to the LSTM model
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience, mode='min'
    )

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping])

    return history


def predict(train, valid, test):
    """
    Creates and trains a model for predicitng the price of btc

    Inputs:
    train: training dataset
    valid: validation dataset
    test: testing dataset
    """
    window = GenerateWindow(24, 1, 1, train, valid, test,
                            ['Close'])

    lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(24, return_sequences=True),
        tf.keras.layers.Dense(units=1)
    ])

    history = create_model_and_compile(lstm_model, window)

    validation_performance = {}
    test_performance = {}

    validation_performance["LSTM"] = lstm_model.evaluate(window.val)
    test_performance["LSTM"] = lstm_model.evaluate(window.test, verbose=0)

    print(validation_performance)
    print(test_performance)
    window.plot(lstm_model)


if __name__ == "__main__":
    from preprocess_data import preprocess
    train_data, valid_data, test_data = preprocess()
    predict(train_data, valid_data, test_data)
