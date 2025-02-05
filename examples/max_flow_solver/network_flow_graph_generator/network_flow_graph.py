import random
import networkx as nx

from typing import List

"""
The Algorithm for the automatic generation of Network Flow Graphs was derived from: 

Deaconu, A. M. & Spridon, D. (2021).
Adaptation of Random Binomial Graphs for Testing Network Flow Problems Algorithms. Mathematics, 9(15), 1716.
https://doi.org/10.3390/math9151716
"""


class Arc:
    def __init__(self, i, u, c):
        self.i = i  # End node of the arc
        self.u = u  # Capacity of the arc
        self.c = c  # Cost of the arc

    def __str__(self):
        return f"(i: {self.i}, u: {self.u}, c: {self.c})"

    def copy(self):
        return Arc(self.i, self.u, self.c)


class NetworkFlowGraph:
    def __init__(
        self,
        p: float,
        nnod: int,
        npath: int,
        ncycle: int,
        mincap: int,
        maxcap: int,
        mincost: int,
        maxcost: int,
    ):
        """
        Initializes the network flow graph with nodes, paths, cycles, capacities, and costs.

        Parameters
        ----------
        p : float
            Probability of adding random edges.
        nnod : int
            Number of nodes in the graph.
        npath : int
            Number of random paths to generate.
        ncycle : int
            Number of random cycles to generate.
        mincap : int
            Minimum capacity for the arcs.
        maxcap : int
            Maximum capacity for the arcs.
        mincost : int
            Minimum cost for the arcs.
        maxcost : int
            Maximum cost for the arcs.

        Returns
        -------
        None
        """
        self.nnod = nnod
        self._adj_list = [[] for _ in range(nnod)]
        self._adj_matrix = [[0] * nnod for _ in range(nnod)]
        self._capacity_matrix = [[0] * nnod for _ in range(nnod)]

        random.seed()

        s = 0
        t = nnod - 1

        for _ in range(npath):
            generate = list(range(1, nnod))
            random.shuffle(generate)

            lpath = random.randint(1, max(1, nnod - 1))
            self._adj_matrix[s][generate[0]] = 1
            for k in range(lpath - 1):
                self._adj_matrix[generate[k]][generate[k + 1]] = 1
            self._adj_matrix[generate[lpath - 1]][t] = 1

        for _ in range(ncycle):
            generate = list(range(nnod))
            random.shuffle(generate)

            lcycle = random.randint(1, nnod)
            for k in range(lcycle - 1):
                if generate[k] != generate[k + 1]:
                    self._adj_matrix[generate[k]][generate[k + 1]] = 1
            if generate[0] != generate[lcycle - 1]:
                self._adj_matrix[generate[lcycle - 1]][generate[0]] = 1

        self._create_la_from_ma(p, mincap, maxcap, mincost, maxcost)

    def _create_la_from_ma(
        self, p: float, mincap: int, maxcap: int, mincost: int, maxcost: int
    ):
        for i in range(self.nnod):
            for j in range(self.nnod):
                if i != j and self._adj_matrix[i][j] == 0 and random.random() < p:
                    self._adj_matrix[i][j] = 1
                if self._adj_matrix[i][j] == 1 and i != j:
                    u = random.randint(mincap, maxcap)
                    c = random.randint(mincost, maxcost)
                    self._adj_list[i].append(Arc(j, u, c))
                    self._capacity_matrix[i][j] = u

    def get_adjacency_list(self) -> List[List[Arc]]:
        """
        Returns the adjacency list of the graph.

        Parameters
        ----------
        None

        Returns
        -------
        List[List[Arc]]
            The adjacency list representing the graph.
        """
        return self._adj_list

    def get_capacity_matrix(self) -> List[List[int]]:
        """
        Returns the capacity matrix of the graph.

        Parameters
        ----------
        None

        Returns
        -------
        List[List[int]]
            The capacity matrix representing the graph.
        """
        return self._capacity_matrix

    def get_adjacency_matrix(self) -> List[List[int]]:
        """
        Returns the adjacency matrix of the graph.

        Parameters
        ----------
        None

        Returns
        -------
        List[List[int]]
            The adjacency matrix representing the graph.
        """
        return self._adj_matrix

    def copy(self):
        """
        Creates a copy of the current NetworkFlowGraph instance.

        Returns
        -------
        NetworkFlowGraph
            A new instance of NetworkFlowGraph with the same properties.
        """
        new_instance = NetworkFlowGraph(0, self.nnod, 0, 0, 0, 0, 0, 0)

        new_instance._adj_matrix = [row[:] for row in self._adj_matrix]
        new_instance._capacity_matrix = [row[:] for row in self._capacity_matrix]
        new_instance._adj_list = [
            [arc.copy() for arc in arcs] for arcs in self._adj_list
        ]

        return new_instance

    def get_inverted_and_reversed_capacity_matrix(self):
        """
        Inverts and reverses the capacity matrix.

        Returns
        -------
        self : NetworkFlowGraph
            The instance with the inverted and reversed capacity matrix.
        """
        n = len(self._capacity_matrix)

        inverted_reversed_matrix = [[0] * n for _ in range(n)]

        # Invert the capacity matrix
        for i in range(n):
            for j in range(n):
                inverted_reversed_matrix[j][i] = self._capacity_matrix[i][j]

        # Reverse the order of the rows and columns
        final_matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                final_matrix[i][j] = inverted_reversed_matrix[n - 1 - i][n - 1 - j]

        self._capacity_matrix = final_matrix

        return self

    def add_path_s_t(self, nonzero_capacity: int):
        """
        Adds a specified nonzero capacity directly between the source (s) and sink (t).

        Parameters
        ----------
        capacity_matrix : List[List[int]]
            The capacity matrix to modify.
        nonzero_capacity : int
            The nonzero capacity to add between source and sink.

        Returns
        -------
        List[List[int]]
            The updated capacity matrix.
        """
        self._capacity_matrix[0][-1] += nonzero_capacity
        return self

    def remove_node(self):
        """
        Removes a random node (excluding source and sink) from the capacity matrix.

        Returns
        -------
        self : NetworkFlowGraph
            The instance with the specified node removed from the capacity matrix.
        """
        n = len(self._capacity_matrix)

        # Exclude source (0) and sink (n - 1) from removal
        node_to_remove = random.choice(range(1, n - 1))

        new_capacity_matrix = [[0] * (n - 1) for _ in range(n - 1)]

        row_shift = 0
        for row in range(n):
            if row == node_to_remove:
                # row_shift set to -1 to skip the node in new matrix
                row_shift = -1
                continue
            col_shift = 0
            for col in range(n):
                if col == node_to_remove:
                    # same as row_shift
                    col_shift = -1
                    continue
                # creates new_capacity_matrix excluding the random node we picked
                new_capacity_matrix[row + row_shift][col + col_shift] = (
                    self._capacity_matrix[row][col]
                )

        self._capacity_matrix = new_capacity_matrix

        return self

    def remove_edge(self):
        """
        Removes a random edge from the capacity matrix.

        Returns
        -------
        self : NetworkFlowGraph
            The instance with the specified edge removed from the capacity matrix.
        """
        n = len(self._capacity_matrix)

        # Choose a random edge to remove
        while True:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j and self._capacity_matrix[i][j] > 0:
                break

        self._capacity_matrix[i][j] = 0

        return self

    def scale_capacities_params(self, scalar: float):
        """
        Scales the Capacities in the Capacity Matrix by the scalar.

        Parameters
        ----------
        scalar : float
            The scalar value to multiply each capacity in the matrix.

        Returns
        -------
        self : NetworkFlowGraph
            The instance with scaled capacity matrix.
        """
        n = len(self._capacity_matrix)
        for i in range(n):
            for j in range(n):
                self._capacity_matrix[i][j] *= scalar

        return self

    def add_node_along_path(
        self, bottleneck: bool = False, same_capacity: bool = False
    ):
        """
        Adds a node along a path in the capacity matrix.

        Parameters
        ----------
        bottleneck : bool, optional
            If True, sets the capacity of the new node connections to 1 (default is False).
        same_capacity : bool, optional
            If True, sets the capacity of the new node connections to the capacity of the split edge (default is False).

        Returns
        -------
        self : NetworkFlowGraph
            The instance with the added node along the path.
        """
        n = len(self._capacity_matrix)

        path = self.find_path(0, n - 1)

        # Pick a random edge along the path
        edge_index = random.randint(0, len(path) - 2)
        i = path[edge_index]
        j = path[edge_index + 1]

        # Find the capacity of the edge being split
        capacity = self._capacity_matrix[i][j]

        # Shift all rows (for correct indexing)
        for row in range(n):
            self._capacity_matrix[row].insert(j, 0)

        self._capacity_matrix.insert(j, [0] * (n + 1))

        if bottleneck:
            new_capacity = 1
        elif same_capacity:
            new_capacity = capacity

        if i < j:
            # Case when the new node is inserted between i and j, and i < j
            self._capacity_matrix[i][j] = new_capacity
            self._capacity_matrix[i][j + 1] = 0
            self._capacity_matrix[j][j + 1] = new_capacity
        else:
            # Case when the new node is inserted between i and j, and i > j
            self._capacity_matrix[i + 1][
                j + 1
            ] = 0  # Old i to old j, now i+1 to j+1, this edge is deleted
            self._capacity_matrix[i + 1][j] = new_capacity  # New edge i+1 -> j
            self._capacity_matrix[j][j + 1] = new_capacity  # New edge j -> j+1

        return self

    def find_path(
        self, source: int, sink: int, path: List[int] = None, visited: set = None
    ):
        """
        Finds a path from the source to the sink in the capacity matrix using depth-first search (DFS).

        Parameters
        ----------
        source : int
            The source node.
        sink : int
            The sink node.
        path : List[int], optional
            The current path being explored (default is None).
        visited : Set[int], optional
            The set of visited nodes (default is None).

        Returns
        -------
        List[int] or None
            The path from source to sink if found, None otherwise.
        """
        if path is None:
            path = []
        if visited is None:
            visited = set()

        path = path + [source]
        visited.add(source)

        if source == sink:
            return path

        for neighbor, capacity in enumerate(self._capacity_matrix[source]):
            if capacity > 0 and neighbor not in visited:
                new_path = self.find_path(neighbor, sink, path, visited)
                if new_path:
                    return new_path

        return None

    def build_graph_from_matrix(self):
        """
        Builds a NetworkX directed graph from the capacity matrix.

        Returns
        -------
        G : networkx.DiGraph
            The directed graph with edges representing the capacities.
        """
        G = nx.DiGraph()
        n = len(self._capacity_matrix)
        for i in range(n):
            for j in range(n):
                if self._capacity_matrix[i][j] >= 0:
                    G.add_edge(i, j, capacity=self._capacity_matrix[i][j])
        return G

    def compute_maxflow(self, maxflow_function):
        """
        Computes the maximum flow using the specified maximum flow solver.

        Parameters
        ----------
        maxflow_function : function
            The maximum flow function to be used for the computation. Solver should take the
            capacity matrix, the source and the sink as arguments.

        Returns
        -------
        int
            The value of the maximum flow from the source to the sink.
        """
        cm = self.get_capacity_matrix()
        source = 0
        sink = len(cm) - 1
        return maxflow_function(cm, source, sink)

    def compute_networkx_flow(self, flow_func):
        """
        Computes the maximum flow using the specified flow function from the NetworkX package.

        Parameters
        ----------
        flow_func : function
            The flow function to be used for the computation. This should be one of the functions
            from `nx.algorithms.flow`.

        Returns
        -------
        int
            The value of the maximum flow.
        """
        G = self.build_graph_from_matrix()
        source, sink = 0, len(self._capacity_matrix) - 1

        flow_value, _ = nx.maximum_flow(G, source, sink, flow_func=flow_func)
        return flow_value


    def compute_networkx_cut(self, flow_func):
        """
        Computes the minimum cut using the specified flow function from the NetworkX package.

        Parameters
        ----------
        flow_func : function
            The flow function to be used for the computation. This should be one of the functions
            from `nx.algorithms.flow`.

        Returns
        -------
        int
            The value of the minimum cut.
        """
        G = self.build_graph_from_matrix()
        source, sink = 0, len(self._capacity_matrix) - 1

        cut_value, _ = nx.minimum_cut(G, source, sink, flow_func=flow_func)
        return cut_value
