#!/usr/bin/env python3
""" Creates the Yolo class for object detection"""

import tensorflow.keras as K
import numpy as np
import cv2
import glob


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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters boxes based on their probability and class confidence

        Inputs:
        boxes: list of numpy.ndarrays containing the processed boundary boxes
            for each output, respectively
        box_confidences: list of numpy.ndarrays containing the processed
            box confidences for each output, respectively
        box_class_probs: list of numpy.ndarrays containing the processed box
            class probabilities for each output, respectively

        Returns:
        tuple of (filtered_boxes, box_classes, box_scores)
        filtered_boxes: numpy.ndarray of shape (?, 4) containing all of the
            filtered bounding boxes
        box_classes: numpy.ndarray of shape (?,) containing the class number
            that each box in filtered_boxes predicts, respectively
        box_scores: numpy.ndarray of shape (?) containing the box scores for
            each box in filtered_boxes, respectively
        """

        filtered_boxes, box_classes, box_scores = [], [], []

        for i in range(len(boxes)):
            box_confidence = box_confidences[i].squeeze(axis=-1)
            box_class_prob = box_class_probs[i]

            # Find the class with the maximum box score
            box_score = box_confidence * box_class_prob
            box_class = np.argmax(box_score, axis=-1)

            box_score = np.max(box_score, axis=-1)

            mask = box_score >= self.class_t

            filtered_boxes += boxes[i][mask].tolist()

            box_classes += box_class[mask].tolist()

            box_scores += box_score[mask].tolist()

        filtered_boxes = np.array(filtered_boxes)
        box_classes = np.array(box_classes)
        box_scores = np.array(box_scores)

        return filtered_boxes, box_classes, box_scores

    def _iou(self, boxes, thresh, scores):
        """
        Calculates the Intersection over Union (IoU) of two bounding boxes

        Inputs:
        boxes: numpy.ndarray of shape (n, 4) containing the boundary boxes
        thresh: threshold for the IoU
        scores: numpy.ndarray of shape (n,) containing the box scores

        Returns:
        numpy.ndarray of shape (m,) containing the indices of the boxes to keep
        """

        # Extract the coordinates of all boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # Calculate the area of each box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        # Sort the indices of boxes by descending scores
        idxs = scores.argsort()[::-1]

        pick = []

        while len(idxs) > 0:
            # Take the index of the top-ranked box and remove it from idxs
            i = idxs[0]
            j = idxs[1:]

            # Add the top-ranked box to the list of picks
            pick.append(i)

            # Find the coordinates of the intersection rectangle
            xx1 = np.maximum(x1[i], x1[j])
            yy1 = np.maximum(y1[i], y1[j])
            xx2 = np.minimum(x2[i], x2[j])
            yy2 = np.minimum(y2[i], y2[j])

            # Compute the width and height of the intersection rectangle
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # Calculate intersection area and union area
            intersection = w * h
            union = area[i] + area[j] - intersection

            # Compute IoU
            iou = intersection / union

            # Select indices where IoU is less than the threshold to keep
            idxs = idxs[np.where(iou <= thresh)[0] + 1]

        # Return the indices of the boxes to keep
        return np.array(pick)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Suppresses all non-max filter boxes

        Inputs:
        filtered_boxes: numpy.ndarray of shape (?, 4) containing all of the
            filtered bounding boxes
        box_classes: numpy.ndarray of shape (?,) containing the class number
            that each box in filtered_boxes predicts, respectively
        box_scores: numpy.ndarray of shape (?) containing the box scores for
            each box in filtered_boxes, respectively

        Returns:
        tuple of (box_predictions, predicted_box_classes, predicted_box_scores)
        box_predictions: numpy.ndarray of shape (?, 4) containing all of the
            predicted bounding boxes ordered by class and box score
        predicted_box_classes: numpy.ndarray of shape (?,) containing the class
            number for box_predictions ordered by class and box score,
            respectively
        predicted_box_scores: numpy.ndarray of shape (?) containing the box
            scores for box_predictions ordered by class and box score,
            respectively
        """

        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        # Process each class separately
        for c in set(box_classes):
            # Find indices of all boxes belonging to the current class
            idx = np.where(box_classes == c)

            # Extract boxes, scores, and classes for the current class
            filtered_boxes_c = filtered_boxes[idx]
            box_scores_c = box_scores[idx]
            box_classes_c = box_classes[idx]

            # Apply IoU-based filtering
            pick = self._iou(filtered_boxes_c, self.nms_t, box_scores_c)

            box_predictions += filtered_boxes_c[pick].tolist()
            predicted_box_classes += box_classes_c[pick].tolist()
            predicted_box_scores += box_scores_c[pick].tolist()

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """
        Loads images from a folder

        Inputs:
        folder_path: string representing the path to the folder holding all
            the images to load

        Returns:
        tuple of (images, image_paths)
            images: list of images as numpy.ndarrays
            image_paths: list of paths to the individual images in images
        """
        images = []
        image_paths = []

        for img_path in glob.glob(folder_path + '/*'):
            image = cv2.imread(img_path)
            images.append(image)
            image_paths.append(img_path)

        return images, image_paths

    def preprocess_images(self, images):
        """
        Resizes and rescales the images before feeding them into the CNN

        Inputs:
        images: list of images as numpy.ndarrays

        Returns:
        tuple of (pimages, image_shapes)
            pimages: numpy.ndarray of shape (ni, input_h, input_w, 3)
            containing
                all of the preprocessed images
                ni: number of images that were preprocessed
                input_h: the input height for the Darknet model
                input_w: the input width for the Darknet model
                3: number of color channels
            image_shapes: numpy.ndarray of shape (ni, 2) containing
                the original height and width of the images
        """
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for image in images:
            image_shapes.append(image.shape[:2])

            # Resize and rescale the images
            pimage = cv2.resize(image, (input_w, input_h),
                                interpolation=cv2.INTER_CUBIC) / 255

            pimages.append(pimage)

        return np.array(pimages), np.array(image_shapes)
    
    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Displays the image with all boundary boxes, class names, and box scores

        Inputs:
        image: numpy.ndarray of shape (h, w, 3) containing the image to display
        boxes: numpy.ndarray of shape (n, 4) containing the boundary boxes
        box_classes: numpy.ndarray of shape (n,) containing the class number
            that each box in boxes predicts, respectively
        box_scores: numpy.ndarray of shape (n,) containing the box scores for
            each box in boxes, respectively
        file_name: file name for the displayed image
        """
        for i in range(len(boxes)):
            box = boxes[i]
            box_class = self.class_names[box_classes[i]]
            box_score = (box_scores[i])

            start_box = (int(box[0]), int(box[1]))
            end_box = (int(box[2]), int(box[3]))
            text_loc = (int(box[0]), int(box[1]) - 5)
            box_color = (255, 0, 0)
            text_color = (0, 0, 255)
            thickness = 1

            cv2.rectangle(image, start_box, end_box, box_color, thickness)
            text = "{} {:.2f}".format(box_class, box_score)
            cv2.putText(image, text, text_loc,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, thickness)

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)

        if key == ord('s'):
            cv2.imwrite(file_name, image)

        cv2.destroyAllWindows()
        return key
