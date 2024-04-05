#!/usr/bin/env python3
""" Creates the Yolo class for object detection"""

import tensorflow.keras as K
import numpy as np


class Yolo:
    """ Uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializer Yolo Class

        Inputs:
        model_path: path to where a Darknet Keras model is stored
        classes_path: path to where the list of class names used for the
            Darknet model, listed in order of index, can be found
        class_t: float representing the box score threshold for the initial
            filtering step
        nms_t: float representing the IOU threshold for non-max suppression
        anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
            containing all of the anchor boxes:
            outputs: number of outputs (predictions) made by the Darknet model
            anchor_boxes: number of anchor boxes used for each prediction
            2: [anchor_box_width, anchor_box_height]
        """
        # Load model
        self.model = K.models.load_model(model_path)

        # Load classes
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]

        # Public instance attributes
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """
        Sigmoid function

        Inputs:
        x: value to transform with sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Outputs is a list of numpy.ndarrays containing the predictions from
        the Darknet model for a single image

        Inputs:
        outputs: list of numpy.ndarrays containing the predictions from the
            Darknet model for a single image
        image_size: numpy.ndarray containing the image’s original size

        Returns:
        tuple of (boxes, box_confidences, box_class_probs)
        boxes: list of numpy.ndarrays containing the processed boundary boxes
            for each output, respectively
        box_confidences: list of numpy.ndarrays containing the processed
            box confidences for each output, respectively
        box_class_probs: list of numpy.ndarrays containing the processed box
            class probabilities for each output, respectively
        """

        boxes, box_confidences, box_class_probs = [], [], []

        for i, output in enumerate(outputs):
            # Image parameters
            grid_h, grid_w = output.shape[:2]
            anchors = self.anchors[i]

            # Center coordinates of boxes
            txy = output[..., :2]
            twh = output[..., 2:4]

            # Use sigmoid to find probability of objects
            conf_sigmoid = self.sigmoid(output[..., 4:5])
            prob_sigmoid = self.sigmoid(output[..., 5:])

            conf_box = np.expand_dims(conf_sigmoid, axis=-1)

            box_confidences.append(conf_box)
            box_class_probs.append(prob_sigmoid)

            # Find actual width and height of boxes
            box_wh = anchors * np.exp(twh)
            box_wh /= [self.model.input[0].shape[1],
                       self.model.input[0].shape[2]]

            # Use grid to find actual location of the image
            grid = np.tile(np.indices((grid_w, grid_h)).T,
                           anchors.shape[0]).reshape(grid_h, grid_w, -1, 2)

            # Find edges of boxes
            box_xy = (self.sigmoid(txy) + grid) / [grid_w, grid_h]
            box_xy1 = box_xy - (box_wh / 2)
            box_xy2 = box_xy + (box_wh / 2)

            box = np.concatenate((box_xy1, box_xy2), axis=-1)
            box *= np.tile(image_size, 2)

            boxes.append(box)

        return boxes, box_confidences, box_class_probs
