from utils.substrate_generator import generate_substrate_network

G = generate_substrate_network()

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

for n, data in G.nodes(data=True):
    print(f"Node {n}: CPU={data['cpu']}")

for u, v, data in G.edges(data=True):
    print(f"Link ({u}, {v}): BW={data['bandwidth']}")
