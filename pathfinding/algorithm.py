import heapq


class PathfindingAlgorithm:
    def __init__(self, graph):
        self.graph = graph

    def find_path(self, start_node, end_node):
        """
        Find the optimal path from start_node to end_node using a pathfinding algorithm.

        :param start_node: The starting node.
        :param end_node: The destination node.
        :return: A list of nodes representing the optimal path.
        """
        # Create an empty dictionary to keep track of the cost of visiting each node in the graph.
        # Initialize the cost of the starting node to zero and the costs of all other nodes to infinity.
        costs = {node: float("inf") for node in self.graph.nodes}
        costs[start_node] = 0

        # Create an empty dictionary to keep track of the optimal path to each node in the graph.
        # Initialize the optimal path to the starting node to an empty list.
        paths = {node: [] for node in self.graph.nodes}
        paths[start_node] = [start_node]

        # Create an empty priority queue to keep track of the nodes to visit.
        queue = []
        heapq.heappush(queue, (0, start_node))

        # Visit each node in the priority queue.
        while queue:
            # Get the node with the lowest cost from the priority queue.
            current_cost, current_node = heapq.heappop(queue)

            # If we have already found a lower cost to the current node, skip it.
            if current_cost > costs[current_node]:
                continue

            # Visit each neighbor of the current node.
            for neighbor in self.graph.neighbors(current_node):
                # Calculate the cost of visiting the neighbor.
                # This is the sum of the cost of visiting the current node and the cost of the edge between the
                # current node and the neighbor.
                cost = current_cost + self.graph.cost(current_node, neighbor)

                # If we have found a lower cost to the neighbor, update the costs and optimal paths.
                if cost < costs[neighbor]:
                    costs[neighbor] = cost
                    paths[neighbor] = paths[current_node] + [neighbor]

                    # Add the neighbor to the priority queue.
                    heapq.heappush(queue, (cost, neighbor))

        # Return the optimal path to the destination node.
        return paths[end_node]
