import numpy as np
from graph.node import Node
from graph.edge import Edge
from preprocessing.transform import ImageTransformer


class GraphBuilder:
    def __init__(self, image_path, detection_model):
        self.image_path = image_path
        self.transformer = ImageTransformer()
        self.detection_model = detection_model
        self.nodes = []

    def build_graph(self):
        # Preprocess the image and get object detection results
        image = self.transformer.load_image(self.image_path)
        preprocessed_image = self.transformer.preprocess_image(image)
        detection_results = self.detection_model.detect_objects(preprocessed_image)

        # Create a node for each detected object
        for idx, detection in enumerate(detection_results):
            object_class = detection['class']
            object_position = np.array(detection['position'])

            node = Node(idx, object_position)
            self.nodes.append(node)

        # Connect adjacent nodes with edges
        for i, node1 in enumerate(self.nodes):
            for j, node2 in enumerate(self.nodes[i+1:]):
                distance = np.linalg.norm(node1.position - node2.position)
                edge = Edge(node1, node2, distance)

                node1.add_adjacent_node(node2, distance)
                node2.add_adjacent_node(node1, distance)

        return self.nodes
