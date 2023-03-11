from ba_network import ba_network
import networkx as nx

# set parameters
n = 6
m = 2

G = ba_network(n, m)
nx.draw_networkx(G)
