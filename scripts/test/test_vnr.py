from utils.vnr_generator import generate_virtual_network_request

G = generate_virtual_network_request()

print("VNR nodes:", G.number_of_nodes())
print("VNR edges:", G.number_of_edges())

for n, d in G.nodes(data=True):
    print(f"  Node {n}: CPU={d['cpu']}")

for u, v, d in G.edges(data=True):
    print(f"  Link ({u}, {v}): BW={d['bandwidth']}")
