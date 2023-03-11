import networkx as nx
import random


def ra_network(n: int, m: int):
    # check the input values are allowed
    if m >= n or m < 1:
        raise (Exception("Invalid parameters"))

    # the initial graph is a fully connected graph of m+1 nodes
    G = nx.complete_graph(m + 1)

    # create a repeated list of nodes to choose from
    # with each node added degree(node) times
    node_choices = [node for node, _ in G.degree()]

    def get_targets(m: int):
        """
        Get m unique target nodes from node_choices in env
        """
        # a set doesn't allow repeats
        targets = set()
        # stop when we have m targets
        while m > len(targets):
            # pick another target from node_choices
            targets.add(random.choice(node_choices))
        # return the targets
        return targets

    # add the remaining nodes
    current_size = G.number_of_nodes()
    while current_size < n:
        # new node is labelled current size
        new_node = current_size
        # find target nodes for new edges
        target_nodes = get_targets(m)
        # add new node
        G.add_node(new_node)
        # add new edges
        G.add_edges_from([(new_node, target) for target in target_nodes])
        # add the new node to the list of node choices
        node_choices.append(new_node)
        # update current size
        current_size += 1

    return G
