from collections import deque


class SortableDigraph:
    """Base class for sortable directed graphs"""
    def __init__(self):
        self.adjacency_list = {}
    
    def add_node(self, node):
        """Add a node to the graph"""
        if node not in self.adjacency_list:
            self.adjacency_list[node] = []
    
    def add_edge(self, start, end):
        """Add an edge from start to end"""
        if start not in self.adjacency_list:
            self.add_node(start)
        if end not in self.adjacency_list:
            self.add_node(end)
        self.adjacency_list[start].append(end)
    
    def get_nodes(self):
        """Return all nodes in the graph"""
        return list(self.adjacency_list.keys())
    
    def get_neighbors(self, node):
        """Return neighbors of a node"""
        return self.adjacency_list.get(node, [])


class TraversableDigraph(SortableDigraph):
    """Digraph with DFS and BFS traversal capabilities"""
    
    def dfs(self, start_node):
        """
        Perform depth-first search traversal starting from start_node.
        Yields each node as it is visited.
        """
        if start_node not in self.adjacency_list:
            return
        
        visited = set()
        stack = [start_node]
        
        while stack:
            node = stack.pop()
            
            if node not in visited:
                visited.add(node)
                yield node
                
                # Add neighbors to stack in reverse order to maintain left-to-right traversal
                neighbors = self.get_neighbors(node)
                for neighbor in reversed(neighbors):
                    if neighbor not in visited:
                        stack.append(neighbor)
    
    def bfs(self, start_node):
        """
        Perform breadth-first search traversal starting from start_node.
        Yields each node as it is visited.
        Uses a deque for efficiency.
        """
        if start_node not in self.adjacency_list:
            return
        
        visited = set()
        queue = deque([start_node])
        visited.add(start_node)
        
        while queue:
            node = queue.popleft()
            yield node
            
            for neighbor in self.get_neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)


class DAG(TraversableDigraph):
    """
    Directed Acyclic Graph - ensures no cycles can be created.
    Inherits from TraversableDigraph.
    """
    
    def add_edge(self, start, end):
        """
        Add an edge from start to end, but only if it doesn't create a cycle.
        Raises ValueError if adding the edge would create a cycle.
        """
        # First add nodes if they don't exist
        if start not in self.adjacency_list:
            self.add_node(start)
        if end not in self.adjacency_list:
            self.add_node(end)
        
        # Check if adding this edge would create a cycle
        # A cycle would exist if there's already a path from end to start
        if self._has_path(end, start):
            raise ValueError(f"Adding edge from '{start}' to '{end}' would create a cycle")
        
        # Safe to add the edge
        self.adjacency_list[start].append(end)
    
    def _has_path(self, start, target):
        """
        Check if there's a path from start to target using DFS.
        Returns True if a path exists, False otherwise.
        """
        if start not in self.adjacency_list or target not in self.adjacency_list:
            return False
        
        if start == target:
            return True
        
        visited = set()
        stack = [start]
        
        while stack:
            node = stack.pop()
            
            if node == target:
                return True
            
            if node not in visited:
                visited.add(node)
                for neighbor in self.get_neighbors(node):
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        return False


# Example usage and testing
if __name__ == "__main__":
    # Create the clothing DAG from the diagram
    clothing_dag = DAG()
    
    # Add edges according to the diagram
    clothing_dag.add_edge("shirt", "pants")
    clothing_dag.add_edge("shirt", "socks")
    clothing_dag.add_edge("shirt", "vest")
    clothing_dag.add_edge("pants", "tie")
    clothing_dag.add_edge("pants", "belt")
    clothing_dag.add_edge("pants", "shoes")
    clothing_dag.add_edge("socks", "shoes")
    clothing_dag.add_edge("tie", "jacket")
    clothing_dag.add_edge("belt", "jacket")
    clothing_dag.add_edge("vest", "jacket")
    clothing_dag.add_edge("shoes", "jacket")
    
    print("DFS traversal starting from 'shirt':")
    for node in clothing_dag.dfs("shirt"):
        print(f"  {node}")
    
    print("\nBFS traversal starting from 'shirt':")
    for node in clothing_dag.bfs("shirt"):
        print(f"  {node}")
    
    # Test cycle detection
    print("\nTesting cycle detection:")
    try:
        clothing_dag.add_edge("jacket", "shirt")
        print("  ERROR: Should have raised an exception!")
    except ValueError as e:
        print(f"  Successfully caught cycle: {e}")
    
    # Test a valid edge that doesn't create a cycle
    try:
        # This should work - no cycle created
        test_dag = DAG()
        test_dag.add_edge("A", "B")
        test_dag.add_edge("B", "C")
        test_dag.add_edge("A", "C")  # This is fine - just creates multiple paths
        print("  Successfully added edge A->C (no cycle)")
    except ValueError as e:
        print(f"  Unexpected error: {e}")
