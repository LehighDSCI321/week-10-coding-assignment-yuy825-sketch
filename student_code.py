"""
TraversableDigraph and DAG module for Week 10.
This module extends SortableDigraph with traversal methods and cycle detection.
"""
from collections import deque


class VersatileDigraph:
    """A class to represent a directed graph with node and edge metadata."""

    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.adjacency = {}

    def add_node(self, node_id, node_value=0):
        """Given a node_id and an optional node_value, add the node to the graph"""
        self.nodes[node_id] = node_value
        if node_id not in self.adjacency:
            self.adjacency[node_id] = []

    def add_edge(
        self,
        start_node,
        end_node,
        start_node_value=0,
        end_node_value=0,
        edge_name="",
        edge_weight=0
    ):
        """Add an edge to the graph with optional parameters"""
        if start_node not in self.nodes:
            self.add_node(start_node, start_node_value)
        if end_node not in self.nodes:
            self.add_node(end_node, end_node_value)

        self.edges[(start_node, end_node)] = {
            "weight": edge_weight, "name": edge_name
        }

        if start_node not in self.adjacency:
            self.adjacency[start_node] = []
        self.adjacency[start_node].append((end_node, edge_name))

    def get_nodes(self):
        """Return a list of nodes in the graph"""
        return list(self.nodes.keys())

    def get_edge_weight(self, start_node, end_node):
        """Given the start_node and the end_node for an edge, return the edge weight"""
        return self.edges.get((start_node, end_node), {}).get("weight", 0)

    def get_node_value(self, node_id):
        """Given a node_id, return the node value"""
        return self.nodes.get(node_id, 0)

    def predecessors(self, target_node):
        """Given a node, return a list of nodes that immediately precede that node"""
        return [
            src for (src, dst) in self.edges
            if dst == target_node
        ]

    def successors(self, src_node):
        """Given a node, return a list of nodes that immediately succeed that node"""
        return [dst for dst, _ in self.adjacency.get(src_node, [])]

    def indegree(self, target_node):
        """Given a node, return the number of edges that lead to that node"""
        return sum(
            1 for (src, dst) in self.edges
            if dst == target_node
        )

    def outdegree(self, src_node):
        """Given a node, return the number of edges that lead from that node"""
        return len(self.adjacency.get(src_node, []))


class SortableDigraph(VersatileDigraph):
    """A directed graph class with topological sorting capability."""

    def top_sort(self):
        """
        Return a topologically sorted list of nodes in the graph.
        Uses Kahn's algorithm (BFS-based approach).
        """
        in_degrees = {n: self.indegree(n) for n in self.nodes}
        queue = [n for n in self.nodes if in_degrees[n] == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            for nxt in self.successors(current):
                in_degrees[nxt] -= 1
                if in_degrees[nxt] == 0:
                    queue.append(nxt)

        if len(result) != len(self.nodes):
            raise ValueError("Graph contains a cycle")

        return result


class TraversableDigraph(SortableDigraph):
    """Extends SortableDigraph with depth-first and breadth-first traversal methods."""

    def dfs(self, start_node):
        """
        Perform depth-first search traversal starting from start_node.
        Yields each node as it is visited (NOT including the start node).
        Uses a stack (implemented with a list).
        """
        if start_node not in self.nodes:
            return

        visited = set([start_node])
        stack = list(self.successors(start_node))[::-1]

        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                yield current
                successors = self.successors(current)
                for nxt in reversed(successors):
                    if nxt not in visited:
                        stack.append(nxt)

    def bfs(self, start_node):
        """
        Perform breadth-first search traversal starting from start_node.
        Yields each node as it is visited (NOT including the start node).
        Uses a deque for efficient FIFO operations.
        """
        if start_node not in self.nodes:
            return

        visited = set([start_node])
        queue = deque([start_node])

        while queue:
            current = queue.popleft()
            for nxt in self.successors(current):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
                    yield nxt


class DAG(TraversableDigraph):
    """Directed Acyclic Graph - ensures no cycles can be created."""

    def add_edge(
        self,
        start_node,
        end_node,
        start_node_value=0,
        end_node_value=0,
        edge_name="",
        edge_weight=0
    ):
        """Add an edge only if it doesn't create a cycle."""
        if start_node not in self.nodes:
            self.add_node(start_node, start_node_value)
        if end_node not in self.nodes:
            self.add_node(end_node, end_node_value)

        if self._has_path_dfs(end_node, start_node):
            raise ValueError(
                f"Adding edge from {start_node} to {end_node} would create a cycle"
            )

        super().add_edge(
            start_node, end_node,
            start_node_value, end_node_value,
            edge_name, edge_weight
        )

    def _has_path_dfs(self, start, target):
        """Return True if a path exists from start to target."""
        if start not in self.nodes or target not in self.nodes:
            return False

        if start == target:
            return True

        visited = set()
        stack = [start]

        while stack:
            current = stack.pop()
            if current == target:
                return True

            if current not in visited:
                visited.add(current)
                for nxt in self.successors(current):
                    if nxt not in visited:
                        stack.append(nxt)

        return False


if __name__ == "__main__":
    print("=== Testing TraversableDigraph ===")
    tg = TraversableDigraph()
    tg.add_edge("A", "B")
    tg.add_edge("A", "C")
    tg.add_edge("B", "D")
    tg.add_edge("C", "D")
    tg.add_edge("D", "E")

    print("\nDFS from A:")
    for n in tg.dfs("A"):
        print(f"  {n}")

    print("\nBFS from A:")
    for n in tg.bfs("A"):
        print(f"  {n}")

    print("\n=== Testing DAG (Clothing Example) ===")
    clothing = DAG()
    clothing.add_edge("shirt", "pants")
    clothing.add_edge("shirt", "socks")
    clothing.add_edge("shirt", "vest")
    clothing.add_edge("pants", "tie")
    clothing.add_edge("pants", "belt")
    clothing.add_edge("pants", "shoes")
    clothing.add_edge("socks", "shoes")
    clothing.add_edge("tie", "jacket")
    clothing.add_edge("belt", "jacket")
    clothing.add_edge("vest", "jacket")
    clothing.add_edge("shoes", "jacket")

    print("\nDFS traversal from 'shirt':")
    for item in clothing.dfs("shirt"):
        print(f"  {item}")

    print("\nBFS traversal from 'shirt':")
    for item in clothing.bfs("shirt"):
        print(f"  {item}")

    print("\n=== Testing Cycle Detection ===")
    try:
        clothing.add_edge("jacket", "shirt")
        print("ERROR: Should have raised an exception!")
    except ValueError as e:
        print(f"✓ Correctly prevented cycle: {e}")

    try:
        test_dag = DAG()
        test_dag.add_edge("A", "B")
        test_dag.add_edge("B", "C")
        test_dag.add_edge("A", "C")
        print("✓ Successfully added A->C (no cycle created)")
    except ValueError as e:
        print(f"ERROR: {e}")

    print("\nTopological sort of clothing DAG:")
    print("  ", clothing.top_sort())
