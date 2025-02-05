import random

from examples.max_flow_solver.network_flow_graph_generator.network_flow_graph import NetworkFlowGraph
from examples.max_flow_solver.solvers.edmonds_karp import max_flow as ek_maxflow
from examples.max_flow_solver.solvers.ford_fulkerson import max_flow as ff_maxflow
from examples.max_flow_solver.solvers.push_relabel import MaxFlow as pr_maxflow
from examples.max_flow_solver.solvers.dinic_algorithm import MaxFlow as din_maxflow

import gemtest as gmt
import networkx as nx

# Create a List of input graphs
n_graphs = 10 # Reasonably large number here, theres 216 Test Cases per Graph 
graphs = []

scalar_dict = {"scalar": [2.0, 7.0, 10.0], "fraction": [0.5, 0.75]}

for _ in range(n_graphs):
    # Generate random parameters for each graph
    p = random.uniform(0.1, 0.6)
    nnod = random.randint(3, 20)
    npath = random.randint(2, 5)
    ncycle = random.randint(0, 3)
    mincap = random.randint(1, 10)
    maxcap = random.randint(10, 20)
    mincost = random.randint(1, 5)
    maxcost = random.randint(5, 20)

    # Create an instance of NetworkCreation
    network_creation = NetworkFlowGraph(p, nnod, npath, ncycle, mincap, maxcap, mincost, maxcost)
    graphs.append(network_creation)

# Define the Relation Names and Datasources
s_t_path = gmt.create_metamorphic_relation(name="s_t_path", data=graphs)
scale_cap = gmt.create_metamorphic_relation(name="scale_cap", data=graphs, parameters=scalar_dict)
scale_cap_ratio = gmt.create_metamorphic_relation(name="scale_cap_ratio", data=graphs, parameters=scalar_dict)
invert_flow = gmt.create_metamorphic_relation(name="invert_flow", data=graphs)
remove_edge = gmt.create_metamorphic_relation(name="remove_edge", data=graphs)
remove_node = gmt.create_metamorphic_relation(name="remove_node", data=graphs)
add_bottleneck = gmt.create_metamorphic_relation(name="add_bottleneck", data=graphs)
add_same_cap_node = gmt.create_metamorphic_relation(name="add_same_cap", data=graphs)

# Define the input transformations
@gmt.transformation(s_t_path)
@gmt.randomized("nonzero_capacity", gmt.RandInt(1, 10))
def add_path_s_t(graph, nonzero_capacity: int):
    """
    Adds a path from s to t if it does not exist.
    Increases the paths capacity if it does exist
    """
    return graph.add_path_s_t(nonzero_capacity)

@gmt.general_transformation(scale_cap)
def scale_capacities_params(mtc: gmt.MetamorphicTestCase):
    """
    Scales the Capacities in the Capacity Matrix by the scalars
    specified in the parameters dicitonary.
    """
    network_graph = mtc.source_input
    scalar = mtc.parameters["scalar"]
    return network_graph.scale_capacities_params(scalar)

@gmt.general_transformation(scale_cap_ratio)
def scale_capacities_params(mtc: gmt.MetamorphicTestCase):
    """
    Scales the Capacities in the Capacity Matrix by the scalars
    specified in the parameters dicitonary.
    """
    network_graph = mtc.source_input
    scalar = mtc.parameters["fraction"]
    return network_graph.scale_capacities_params(scalar)

@gmt.transformation(invert_flow)
def invert_cap_matrix(graph):
    return graph.get_inverted_and_reversed_capacity_matrix()

@gmt.transformation(remove_edge)
def remove_random_edge(graph):
    return graph.remove_edge()

@gmt.transformation(remove_node)
def remove_random_node(graph):
    return graph.remove_node()

@gmt.transformation(add_bottleneck)
def add_bottleneck_along_path(graph):
    return graph.add_node_along_path(bottleneck=True)

@gmt.transformation(add_same_cap_node)
def add_same_cap_along_path(graph):
    return graph.add_node_along_path(same_capacity=True)

# Define the relations
@gmt.general_relation(scale_cap)
def flow_scaled(mtc: gmt.MetamorphicTestCase):
    """
    Verifies that the maximum flow of the follow_up ouput is exactly scaled by the scalar
    used for increasing the capacities.
    """
    return gmt.approximately((mtc.source_output * mtc.parameters["scalar"]), mtc.followup_output)

@gmt.general_relation(scale_cap_ratio)
def flow_fraction(mtc: gmt.MetamorphicTestCase):
    """
    Verifies that the maximum flow of the follow_up ouput is exactly scaled by the scalar
    used for increasing the capacities.
    """
    return gmt.approximately((mtc.source_output * mtc.parameters["fraction"]), mtc.followup_output)

@gmt.relation(s_t_path)
def flow_greater(source_output, followup_output) -> bool:
    """Verifies that the maximum flow of follow-up output is larger"""
    return source_output < followup_output

@gmt.relation(invert_flow, add_same_cap_node)
def flow_equal(source_output, followup_output) -> bool:
    """Verifies that the maximum flow of follow-up output is equal"""
    return gmt.approximately(source_output, followup_output, absolute=1.19e-07)


@gmt.relation(remove_edge, remove_node) 
def flow_leq(source_output, followup_output) -> bool:
    "Verifies that the maximum flow between of follow-up output output is less or equal"
    return source_output >= followup_output

@gmt.relation(add_bottleneck)
def flow_leq_or_one_greater(source_output, followup_output)-> bool:
    return source_output >= followup_output or source_output + 1 == followup_output

@gmt.system_under_test()
def test_find_maximum_flow_ek(capacity_graph):
    return capacity_graph.compute_maxflow(ek_maxflow)

@gmt.system_under_test()
def test_find_maximum_flow_ff(capacity_graph):
    return capacity_graph.compute_maxflow(ff_maxflow)

@gmt.system_under_test()
def test_find_maximum_flow_pr(capacity_graph):
    return capacity_graph.compute_maxflow(pr_maxflow)

@gmt.system_under_test()
def test_find_maximum_flow_dinic(capacity_graph):
    return capacity_graph.compute_maxflow(din_maxflow)

# NetworkX Solvers. Testing Various flow functions
# Also Testing minimum cut function as they are equal to the maximum flow,
# according to the Max-Flow-Min-Cut Theorem

@gmt.system_under_test()
def test_networkx_preflow_push(capacity_graph):
    return capacity_graph.compute_networkx_flow(nx.algorithms.flow.preflow_push)

@gmt.system_under_test()
def test_networkx_boykov_kolmogorov(capacity_graph):
    return capacity_graph.compute_networkx_flow(nx.algorithms.flow.boykov_kolmogorov)

@gmt.system_under_test()
def test_networkx_dinitz(capacity_graph):
    return capacity_graph.compute_networkx_flow(nx.algorithms.flow.dinitz)

@gmt.system_under_test()
def test_networkx_shortest_augmenting_path(capacity_graph):
    return capacity_graph.compute_networkx_flow(nx.algorithms.flow.shortest_augmenting_path)

@gmt.system_under_test()
def test_networkx_preflow_push_min_cut(capacity_graph):
    return capacity_graph.compute_networkx_cut(nx.algorithms.flow.preflow_push)

@gmt.system_under_test()
def test_networkx_boykov_kolmogorov_min_cut(capacity_graph):
    return capacity_graph.compute_networkx_cut(nx.algorithms.flow.boykov_kolmogorov)

@gmt.system_under_test()
def test_networkx_dinitz_min_cut(capacity_graph):
    return capacity_graph.compute_networkx_cut(nx.algorithms.flow.dinitz)

@gmt.system_under_test()
def test_networkx_shortest_augmenting_path_min_cut(capacity_graph):
    return capacity_graph.compute_networkx_cut(nx.algorithms.flow.shortest_augmenting_path)
