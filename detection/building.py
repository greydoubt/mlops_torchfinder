import tensorflow as tf
import numpy as np
from detection.model import ObjectDetectionModel
from detection.utils import load_label_map, draw_boxes_and_labels


class BuildingDetector(ObjectDetectionModel):
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

        # Filter out low-confidence detections
        threshold = 0.5
        detection_mask = detection_scores >= threshold
        detection_classes = detection_classes[detection_mask]
        detection_boxes = detection_boxes[detection_mask]

        # Convert detection boxes to positions
        positions = []
        for box in detection_boxes:
            ymin, xmin, ymax, xmax = box
            y = ymin + (ymax - ymin) / 2
            x = xmin + (xmax - xmin) / 2
            positions.append([y, x])

        # Convert detection classes to labels
        label_map = load_label_map(self.label_map_path)
        labels = [label_map[c] for c in detection_classes]

        # Draw bounding boxes on the image for visualization
        image_with_boxes = draw_boxes_and_labels(image, detection_boxes, labels)

        # Return a list of objects with positions and labels
        objects = [{'position': pos, 'class': label} for pos, label in zip(positions, labels)]
        return objects, image_with_boxes
