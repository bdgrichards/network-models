from ra_network import ra_network
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def test_1():
    """
    Check the number of nodes is correct    
    """
    n = 10000
    m = 4
    G = ra_network(n, m)
    print("Expected:", n)
    print("Measured:", G.number_of_nodes())


def test_2():
    """
    Check the number of edges. Note that the 
    initial graph is complete, so we expect m*n - m(m+1)/2
    """
    n = 10000
    m = 4
    G = ra_network(n, m)
    print("Expected:", n*m - int(m*(m+1)/2))
    print("Measured:", G.number_of_edges())


def test_3():
    """
    Check average degree is 2m
    """
    n = 10000
    m = 4
    G = ra_network(n, m)
    print("Expected:", 2*m)
    print("Measured:", np.mean([d for _, d in G.degree()]))


def test_4():
    """
    Check there are 0 nodes with degree less than m
    """
    n = 10000
    m = 4
    G = ra_network(n, m)
    print("Expected: 0")
    print("Measured:", sum([d for _, d in G.degree() if d < m]))


def test_5():
    """
    Draw the graph, in order to check there are no obvious errors
    """
    n = 10
    m = 2
    G = ra_network(n, m)
    nx.draw_networkx(G)
    plt.show()


def test_6():
    """
    Check invalid m isn't allowed
    """
    G = ra_network(10, 0)


def test_7():
    """
    Check invalid m and n combinations aren't allowed
    """
    G = ra_network(5, 5)
