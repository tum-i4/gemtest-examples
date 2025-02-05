import itertools

from dijkstar import Graph, find_path  # type: ignore

import gemtest as gmt

"""
This example demonstrates two MR test of a shortest path algorithm of an undirected graph.
- If the start and end nodes are swapped, the cost of the shortest path should remain the same.
- If an edge that is much cheaper than any other is added between two nodes / overwrites an
 existing edge, the cost of the shortest path should be less or equal than before.
"""

# setup
graph1: Graph = Graph(undirected=True)
graph1.add_edge(1, 2, 10)
graph1.add_edge(2, 3, 15)
graph1.add_edge(3, 4, 18)
graph2: Graph = Graph(undirected=True)
graph2.add_edge(1, 2, 10)
graph2.add_edge(1, 3, 15)
graph2.add_edge(2, 4, 18)
graph2.add_edge(3, 4, 17)


def create_graph_start_end(graph: Graph, start: int, end: int) -> dict:
    return {
        "graph": graph,
        "start": start,
        "end": end
    }


def get_graph_start_end(graph_start_end: dict) -> tuple:
    return graph_start_end["graph"], graph_start_end["start"], graph_start_end["end"]


graphs = [graph1, graph2]
starts = [1, 2]
ends = [3, 4]

permutations = list(itertools.product(graphs, starts, ends))
data = [create_graph_start_end(graph, start, end) for graph, start, end in permutations]

start_end = gmt.create_metamorphic_relation(name="start_end", data=data)
random_cheap = gmt.create_metamorphic_relation(name="random_cheap", data=data)


@gmt.transformation(start_end)
def switch_startend(source_input: dict) -> dict:
    """Switch the starting and destination node from the tuple of pathfinder's inputs."""
    graph, new_end, new_start = get_graph_start_end(source_input)
    return create_graph_start_end(graph, new_start, new_end)


@gmt.transformation(random_cheap)
@gmt.randomized("newedge_start", gmt.RandInt(1, 4))
@gmt.randomized("newedge_end", gmt.RandInt(1, 4))
def add_random_cheap_edge(source_input: dict, newedge_start: int, newedge_end: int) -> dict:
    """Adds an edge that is much cheaper than any other / overwrites an
    existing edge with a much cheaper one."""
    graph, start, end = get_graph_start_end(source_input)
    graph_new = Graph(data=graph.get_data(), undirected=True)
    graph_new.add_edge(newedge_start, newedge_end, 1)
    return create_graph_start_end(graph_new, start, end)


@gmt.relation(start_end)
def cost_equal(source_output, followup_output) -> bool:
    """Verifies that two paths has the same total cost."""
    return source_output == followup_output


@gmt.relation(random_cheap)
def cost_greater_equal(source_output, followup_output) -> bool:
    """Verifies that the first path has greater or equal total cost than the second one."""
    return source_output >= followup_output


@gmt.system_under_test()
def test_find_shortest_path(input_graph: dict):
    """Find a shortest path between two nodes in a graph"""
    graph, start, end = get_graph_start_end(input_graph)
    shortest_path_cost = find_path(graph, start, end).total_cost
    return shortest_path_cost
