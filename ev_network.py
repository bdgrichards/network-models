import networkx as nx
import random


def ev_network(n: int, m: int, r: int):
    # check the input values are allowed
    if m >= n or m < 1 or r > m or r < 1:
        raise (Exception("Invalid parameters"))

    # the initial graph can no longer be complete
    # since adding edges between existing nodes
    G = nx.star_graph(m)

    # create repeated lists of nodes to choose from
    # one for preferential attachment, one for BA
    ra_node_choices = [node for node, _ in G.degree()]
    pa_node_choices = [node for node, deg in G.degree() for _ in range(deg)]

    def get_targets(count: int, choices: list[int]):
        """
        Get `count` unique target nodes from `choices`
        """
        # a set doesn't allow repeats
        targets = set()
        # stop when we have `count` targets
        while count > len(targets):
            # pick another target from choices
            targets.add(random.choice(choices))
        # return the targets
        return targets

    def get_target_pairs(count: int, choices: list[int]):
        """
        Get `count` unique pairs of nodes from `choices`
        """
        target_pairs: set[tuple[int, int]] = set()
        # stop when we have `count` pairs
        while count > len(target_pairs):
            # pick a pair from choices
            pair = random.sample(choices, 2)
            # check it isn't already in the graph, and isn't a loop
            if not G.has_edge(pair[0], pair[1]) and not pair[0] == pair[1]:
                # sort it and add it to the set
                target_pairs.add(tuple(sorted(pair)))
        # return the pairs
        return target_pairs

    # add the remaining nodes
    current_size = G.number_of_nodes()
    while current_size < n:
        # new node is labelled current size
        new_node = current_size
        # r edges from new node with random attachment
        ra_target_nodes = get_targets(r, ra_node_choices)
        # (m-r) edges between existing nodes
        pa_target_pairs = get_target_pairs(m-r, pa_node_choices)
        # add new node
        G.add_node(new_node)
        # add new edges
        G.add_edges_from([(new_node, target) for target in ra_target_nodes])
        G.add_edges_from([(start, end) for (start, end) in pa_target_pairs])
        # update the random attachment choices
        ra_node_choices.append(new_node)
        # update the preferential attachment choices
        pa_node_choices.extend(r*[new_node])
        pa_node_choices.extend(ra_target_nodes)
        pa_node_choices.extend([i for (i, _) in pa_target_pairs])
        pa_node_choices.extend([j for (_, j) in pa_target_pairs])
        # update current size
        current_size += 1

    return G
