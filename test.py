import networkx as nx

G = nx.Graph()

G.add_node("one")
G.add_nodes_from([2, 3])
G.add_nodes_from([
    (4, {"colour": "red"}),
    (5, {"colour": "blue"})
])
G.add_node("Best node")

G.add_edge(1, 2)
G.add_edge(4, 5)
G.add_edge(1, "Best node")

print("Number of edges:", G.number_of_edges())
print("Nodes:", G.nodes)
print("Edges:", G.edges)

# print(G)
