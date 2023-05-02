from typing import Dict, List, Tuple
import heapq
import math


class AStar:
    def __init__(self, graph: Dict[int, List[Tuple[int, float]]], h: Dict[int, float]):
        self.graph = graph
        self.h = h

    def shortest_path(self, start: int, end: int) -> Tuple[List[int], float]:
        distances = {node: float('inf') for node in self.graph}
        distances[start] = 0
        visited = set()
        prev = {}

        pq = [(0 + self.h[start], start)]
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
                    heapq.heappush(pq, (new_cost + self.h[neighbor], neighbor))
        return [], float('inf')

    @staticmethod
    def euclidean_distance(node1: Tuple[float, float], node2: Tuple[float, float]) -> float:
        x1, y1 = node1
        x2, y2 = node2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
