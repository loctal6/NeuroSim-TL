import networkx as nx
import torch
from torch_geometric.data import Data
import random

def generate_random_circuit(num_nodes=11):
    G = nx.DiGraph()
    gate_types = ['AND', 'OR', 'NAND', 'NOR', 'XOR', 'NOT', 'INPUT', 'OUTPUT']
    for i in range(num_nodes):
        G.add_node(i, label=random.choice(gate_types))
    for i in range(1, num_nodes):
        G.add_edge(random.randint(0, i-1), i)
    return G

def load_c17_circuit():
    G = nx.DiGraph()
    for node in ["G1", "G2", "G3", "G6", "G7"]:
        G.add_node(node, label="INPUT")
    G.add_node("G10", label="NAND")
    G.add_node("G11", label="NAND")
    G.add_node("G16", label="NAND")
    G.add_node("G19", label="NAND")
    G.add_node("G22", label="OUTPUT")
    G.add_node("G23", label="OUTPUT")
    G.add_edges_from([
        ("G1", "G10"), ("G3", "G10"),
        ("G3", "G11"), ("G6", "G11"),
        ("G2", "G16"), ("G11", "G16"),
        ("G6", "G19"), ("G7", "G19"),
        ("G10", "G22"), ("G16", "G22"),
        ("G16", "G23"), ("G19", "G23")
    ])
    return G

def load_c432_circuit():
    G = nx.DiGraph()
    for node in ["N1", "N4", "N8", "N11", "N14", "N17", "N21"]:
        G.add_node(node, label="INPUT")
    for node in ["N223", "N432"]:
        G.add_node(node, label="OUTPUT")
    G.add_node("N118", label="NAND")
    G.add_node("N119", label="NOR")
    G.add_node("N120", label="AND")
    G.add_node("N121", label="OR")
    G.add_edges_from([
        ("N1", "N118"), ("N4", "N118"),
        ("N8", "N119"), ("N11", "N119"),
        ("N118", "N223"), ("N119", "N223"),
        ("N14", "N120"), ("N17", "N120"),
        ("N21", "N121"), ("N118", "N121"),
        ("N120", "N432"), ("N121", "N432")
    ])
    return G

def convert_to_pyg_data(G, faulty_nodes=None):
    if faulty_nodes is None:
        faulty_nodes = []
    node_mapping = {name: i for i, name in enumerate(G.nodes())}
    edge_index = torch.tensor(
        [[node_mapping[src], node_mapping[dst]] for src, dst in G.edges()],
        dtype=torch.long).t().contiguous()
    gate_to_vec = {
        'AND': [1,0,0,0,0,0,0,0],
        'OR': [0,1,0,0,0,0,0,0],
        'NAND': [0,0,1,0,0,0,0,0],
        'NOR': [0,0,0,1,0,0,0,0],
        'XOR': [0,0,0,0,1,0,0,0],
        'NOT': [0,0,0,0,0,1,0,0],
        'INPUT': [0,0,0,0,0,0,1,0],
        'OUTPUT': [0,0,0,0,0,0,0,1]
    }
    x = torch.tensor([gate_to_vec[G.nodes[n]['label'].upper()] for n in G.nodes()], dtype=torch.float)
    y = torch.tensor([1 if n in faulty_nodes else 0 for n in G.nodes()], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)

# Load simplified version of c880
def load_c880_circuit():
    G = nx.DiGraph()
    for node in ["A1", "A2", "A3", "A4", "A5"]:
        G.add_node(node, label="INPUT")
    G.add_node("G201", label="AND")
    G.add_node("G202", label="OR")
    G.add_node("G203", label="XOR")
    G.add_node("G204", label="NAND")
    for node in ["G201", "G202", "G203", "G204"]:
        G.add_node(f"{node}_OUT", label="OUTPUT")
    G.add_edges_from([
        ("A1", "G201"), ("A2", "G201"),
        ("A3", "G202"), ("A4", "G202"),
        ("G201", "G201_OUT"), ("G202", "G202_OUT"),
        ("A5", "G203"), ("G201", "G203"),
        ("G203", "G203_OUT"), ("G204", "G204_OUT")
    ])
    return G

# Load simplified version of c5315
def load_c5315_circuit():
    G = nx.DiGraph()
    for node in ["I1", "I2", "I3", "I4", "I5", "I6"]:
        G.add_node(node, label="INPUT")
    for i in range(1, 6):
        G.add_node(f"G{i}", label=random.choice(["AND", "OR", "XOR", "NAND", "NOR"]))
    for o in ["O1", "O2", "O3"]:
        G.add_node(o, label="OUTPUT")
    G.add_edges_from([
        ("I1", "G1"), ("I2", "G1"),
        ("I3", "G2"), ("I4", "G2"),
        ("G1", "G3"), ("G2", "G3"),
        ("I5", "G4"), ("G3", "G4"),
        ("G4", "O1"), ("G2", "O2"), ("G5", "O3")
    ])
    return G
