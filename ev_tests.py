from ev_network import ev_network
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def test_1():
    """
    Check the number of nodes is correct    
    """
    n = 1000
    m = 4
    r = 1
    G = ev_network(n, m, r)
    print("Expected:", n)
    print("Measured:", G.number_of_nodes())


def test_2():
    """
    Check the number of edges. Note that the 
    initial graph is a star graph with m+1 nodes, so we expect m(n - m)
    (no dependence on r)
    """
    n = 10000
    m = 9
    r = 3
    G = ev_network(n, m, r)
    print("Expected:", m*(n-m))
    print("Measured:", G.number_of_edges())


def test_3():
    """
    Check average degree is 2m
    """
    n = 10000
    m = 4
    r = 1
    G = ev_network(n, m, r)
    print("Expected:", 2*m)
    print("Measured:", np.mean([d for _, d in G.degree()]))


def test_4():
    """
    Check there are less than m nodes with degree less than r
    """
    n = 10000
    m = 12
    r = 4
    G = ev_network(n, m, r)
    print("Expected: <%i" % m)
    print("Measured:", sum([d for _, d in G.degree() if d < r]))


def test_5():
    """
    Draw the graph, in order to check there are no obvious errors
    """
    n = 20
    m = 4
    r = 1
    G = ev_network(n, m, r)
    nx.draw_networkx(G)
    plt.show()


def test_6():
    """
    Check invalid m isn't allowed
    """
    G = ev_network(10, -1, 0)


def test_7():
    """
    Check invalid m and n combinations aren't allowed
    """
    G = ev_network(5, 5, 2)


def test_8():
    """
    Check invalid m and r combinations aren't allowed
    """
    G = ev_network(10, 5, 0)


def test_9():
    """
    Check invalid m and r combinations aren't allowed
    """
    G = ev_network(10, 5, 6)
