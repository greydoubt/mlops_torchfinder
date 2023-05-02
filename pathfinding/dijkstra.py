from typing import Dict, List, Tuple
import heapq


class Dijkstra:
    def __init__(self, graph: Dict[int, List[Tuple[int, float]]]):
        self.graph = graph

    def shortest_path(self, start: int, end: int) -> Tuple[List[int], float]:
        distances = {node: float('inf') for node in self.graph}
        distances[start] = 0
        visited = set()
        prev = {}

        pq = [(0, start)]
        while pq:
            (cost, current_node) = heapq.heappop(pq)
            if current_node == end:
                path = []
                while current_node in prev:
                    path.append(current_node)
                    current_node = prev[current_node]
                path.append(start)
                path.reverse()
                return path, distances[end]
            if current_node in visited:
                continue
            visited.add(current_node)
            for neighbor, weight in self.graph[current_node]:
                new_cost = distances[current_node] + weight
                if new_cost < distances[neighbor]:
                    prev[neighbor] = current_node
                    distances[neighbor] = new_cost
                    heapq.heappush(pq, (new_cost, neighbor))
        return [], float('inf')
