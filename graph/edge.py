class Edge:
    def __init__(self, source, destination, weight):
        self.source = source
        self.destination = destination
        self.weight = weight

    def __repr__(self):
        return f"Edge from {self.source} to {self.destination} with weight {self.weight}"
