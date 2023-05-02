import tensorflow as tf
import numpy as np
import cv2


def load_label_map(label_map_path):
    # Load the label map from disk
    with open(label_map_path, 'r') as f:
        label_map = {int(line.split(':')[0]) + 1: line.split(':')[1].strip() for line in f.readlines()}

    # Add a background class at index 0
    label_map[0] = 'background'

    return label_map


def draw_boxes_and_labels(image, boxes, classes, scores=None, label_map=None):
    # Draw bounding boxes and class labels on the image
    image_with_boxes = np.copy(image)
    num_boxes = boxes.shape[0]
    for i in range(num_boxes):
        ymin, xmin, ymax, xmax = boxes[i]
        ymin, xmin, ymax, xmax = int(ymin * image.shape[0]), int(xmin * image.shape[1]), int(ymax * image.shape[0]), int(xmax * image.shape[1])
        cv2.rectangle(image_with_boxes, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        if label_map is not None:
            label = label_map[classes[i]]
            score = scores[i] if scores is not None else None
            label_text = f'{label} {score:.2f}' if score is not None else label
            cv2.putText(image_with_boxes, label_text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return image_with_boxes
