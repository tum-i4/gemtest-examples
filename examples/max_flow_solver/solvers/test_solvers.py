import pytest

from examples.max_flow_solver.solvers.edmonds_karp import max_flow as ek_maxflow
from examples.max_flow_solver.solvers.ford_fulkerson import max_flow as ff_maxflow
from examples.max_flow_solver.solvers.push_relabel import MaxFlow as pr_maxflow
from examples.max_flow_solver.solvers.dinic_algorithm import MaxFlow as din_maxflow

@pytest.fixture
def capacity_matrix():
    return [
        [0, 3, 3, 0, 0, 0],
        [0, 0, 2, 3, 0, 0],
        [0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 4, 2],
        [0, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0],
    ]

@pytest.fixture
def source():
    return 0 

@pytest.fixture
def sink():
    return 5 

def test_max_flow_ek(capacity_matrix, source, sink):
    max_flow_value = ek_maxflow(capacity_matrix, source, sink)
    assert max_flow_value == 4

def test_max_flow_ff(capacity_matrix, source, sink):
    max_flow_value = ff_maxflow(capacity_matrix, source, sink)
    assert max_flow_value == 4

def test_max_flow_pr(capacity_matrix, source, sink):
    max_flow_value = pr_maxflow(capacity_matrix, source, sink)
    assert max_flow_value == 4

def test_max_flow_dinic(capacity_matrix, source, sink):
    max_flow_value = din_maxflow(capacity_matrix, source, sink)
    assert max_flow_value == 4