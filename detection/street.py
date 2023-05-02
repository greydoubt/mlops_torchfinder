import tensorflow as tf
import numpy as np
from detection.model import ObjectDetectionModel
from detection.utils import load_label_map, draw_boxes_and_labels


class StreetDetector(ObjectDetectionModel):
    def __init__(self, model_path, label_map_path):
        super().__init__(model_path, label_map_path)

    def detect_objects(self, image):
        # Run object detection on the image
        input_tensor = np.expand_dims(image, 0)
        detections = self.model(input_tensor)

        # Extract relevant information from the detections
        detection_scores = detections['detection_scores'][0].numpy()
        detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)
        detection_boxes = detections['detection_boxes'][0].numpy()

        # Filter out low-confidence detections and non-street classes
        threshold = 0.5
        detection_mask = np.logical_and(detection_scores >= threshold, detection_classes == 1)
        detection_boxes = detection_boxes[detection_mask]

        # Convert detection boxes to edges
        edges = []
        for box in detection_boxes:
            ymin, xmin, ymax, xmax = box
            start = [ymin + (ymax - ymin) / 2, xmin + (xmax - xmin) / 2]
            end = [ymax, xmin + (xmax - xmin) / 2]
            edges.append((start, end))

        # Draw bounding boxes on the image for visualization
        image_with_boxes = draw_boxes_and_labels(image, detection_boxes, ['street'])

        # Return a list of edges
        return edges, image_with_boxes
