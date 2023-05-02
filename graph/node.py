class Node:
    def __init__(self, id, position):
        self.id = id
        self.position = position
        self.adjacent_nodes = {}

    def add_adjacent_node(self, node, weight):
        self.adjacent_nodes[node.id] = (node, weight)

    def remove_adjacent_node(self, node):
        del self.adjacent_nodes[node.id]

    def get_adjacent_nodes(self):
        return self.adjacent_nodes.values()

    def __repr__(self):
        return f"Node {self.id} at position {self.position}"
