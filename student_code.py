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
            'weight': edge_weight, 'name': edge_name
        }
        
        if start_node not in self.adjacency:
            self.adjacency[start_node] = []
        self.adjacency[start_node].append((end_node, edge_name))
    
    def get_nodes(self):
        """Return a list of nodes in the graph"""
        return list(self.nodes.keys())
    
    def get_edge_weight(self, start_node, end_node):
        """Given the start_node and the end_node for an edge, return the edge weight"""
        return self.edges.get((start_node, end_node), {}).get('weight', 0)
    
    def get_node_value(self, node_id):
        """Given a node_id, return the node value"""
        return self.nodes.get(node_id, 0)
    
    def predecessors(self, node):
        """Given a node, return a list of nodes that immediately precede that node"""
        return [
            start_node for (start_node, end_node) in self.edges
            if end_node == node
        ]
    
    def successors(self, node):
        """Given a node, return a list of nodes that immediately succeed that node"""
        return [end_node for end_node, _ in self.adjacency.get(node, [])]
    
    def indegree(self, node):
        """Given a node, return the number of edges that lead to that node"""
        return sum(
            1 for (start_node, end_node) in self.edges
            if end_node == node
        )
    
    def outdegree(self, node):
        """Given a node, return the number of edges that lead from that node"""
        return len(self.adjacency.get(node, []))


class SortableDigraph(VersatileDigraph):
    """A directed graph class with topological sorting capability."""
    
    def top_sort(self):
        """
        Return a topologically sorted list of nodes in the graph.
        Uses Kahn's algorithm (BFS-based approach).
        """
        # Calculate in-degrees for all nodes
        in_degrees = {}
        for node in self.nodes:
            in_degrees[node] = self.indegree(node)
        
        # Find all nodes with in-degree 0
        queue = [node for node in self.nodes if in_degrees[node] == 0]
        result = []
        
        while queue:
            # Remove a node with in-degree 0
            current = queue.pop(0)
            result.append(current)
            
            # For each successor of current node
            for successor in self.successors(current):
                in_degrees[successor] -= 1
                # If in-degree becomes 0, add to queue
                if in_degrees[successor] == 0:
                    queue.append(successor)
        
        # Check if topological sort is possible (no cycles)
        if len(result) != len(self.nodes):
            raise ValueError("Graph contains a cycle")
        
        return result


class TraversableDigraph(SortableDigraph):
    """
    Extends SortableDigraph with depth-first and breadth-first traversal methods.
    """
    
    def dfs(self, start_node):
        """
        Perform depth-first search traversal starting from start_node.
        Yields each node as it is visited (NOT including the start node).
        Uses a stack (implemented with a list).
        """
        if start_node not in self.nodes:
            return

        visited = set([start_node])  # Mark start as visited but don't yield it
        stack = list(self.successors(start_node))[::-1]  # Add successors in reverse order

        while stack:
            current = stack.pop()

            if current not in visited:
                visited.add(current)
                yield current

                # Get successors and add them to stack in reverse order
                successors = self.successors(current)
                for successor in reversed(successors):
                    if successor not in visited:
                        stack.append(successor)
    
    def bfs(self, start_node):
        """
        Perform breadth-first search traversal starting from start_node.
        Yields each node as it is visited (NOT including the start node).
        Uses a deque for efficient FIFO operations.
        """
        if start_node not in self.nodes:
            return

        visited = set([start_node])  # Mark start as visited but don't yield it
        queue = deque([start_node])

        while queue:
            current = queue.popleft()

            # Add all unvisited successors to the queue
            for successor in self.successors(current):
                if successor not in visited:
                    visited.add(successor)
                    queue.append(successor)
                    yield successor  # Yield when adding to queue, not when processing


class DAG(TraversableDigraph):
    """
    Directed Acyclic Graph - ensures no cycles can be created.
    Inherits from TraversableDigraph, which provides DFS and BFS methods.
    """
    
    def add_edge(
        self,
        start_node,
        end_node,
        start_node_value=0,
        end_node_value=0,
        edge_name="",
        edge_weight=0
    ):
        """
        Add an edge from start_node to end_node, but only if it doesn't create a cycle.
        Raises an exception if adding the edge would create a cycle.
        """
        # First ensure both nodes exist
        if start_node not in self.nodes:
            self.add_node(start_node, start_node_value)
        if end_node not in self.nodes:
            self.add_node(end_node, end_node_value)
        
        # Check if adding this edge would create a cycle
        # A cycle exists if there's already a path from end_node to start_node
        if self._has_path_dfs(end_node, start_node):
            raise ValueError(f"Adding edge from {start_node} to {end_node} would create a cycle")
        
        # Safe to add the edge using parent class method
        super().add_edge(start_node, end_node, start_node_value, end_node_value, edge_name, edge_weight)
    
    def _has_path_dfs(self, start, target):
        """
        Check if there's a path from start to target using depth-first search.
        Returns True if a path exists, False otherwise.
        """
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
                for successor in self.successors(current):
                    if successor not in visited:
                        stack.append(successor)

        return False


# Testing code
if __name__ == "__main__":
    print("=== Testing TraversableDigraph ===")
    
    # Create a simple graph for testing
    tg = TraversableDigraph()
    tg.add_edge("A", "B")
    tg.add_edge("A", "C")
    tg.add_edge("B", "D")
    tg.add_edge("C", "D")
    tg.add_edge("D", "E")
    
    print("\nDFS from A:")
    for node in tg.dfs("A"):
        print(f"  {node}")
    
    print("\nBFS from A:")
    for node in tg.bfs("A"):
        print(f"  {node}")
    
    print("\n=== Testing DAG (Clothing Example) ===")
    
    # Create the clothing DAG from the assignment diagram
    clothing = DAG()
    
    # Add edges according to the diagram
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
    
    # Try to create a cycle - should raise an exception
    try:
        clothing.add_edge("jacket", "shirt")
        print("ERROR: Should have raised an exception!")
    except ValueError as e:
        print(f"✓ Correctly prevented cycle: {e}")
    
    # Test that valid edges still work
    try:
        test_dag = DAG()
        test_dag.add_edge("A", "B")
        test_dag.add_edge("B", "C")
        test_dag.add_edge("A", "C")  # Multiple paths OK, no cycle
        print("✓ Successfully added A->C (no cycle created)")
    except ValueError as e:
        print(f"ERROR: {e}")
    
    # Test topological sort still works
    print("\nTopological sort of clothing DAG:")
    print("  ", clothing.top_sort())
