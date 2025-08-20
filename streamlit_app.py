import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import torch
from model import GCN
from data_generator import load_c17_circuit, load_c432_circuit, load_c880_circuit, load_c5315_circuit, generate_random_circuit, convert_to_pyg_data

st.title("🔍 GNN Fault Localization in VLSI Circuits")
circuit_type = st.selectbox("Choose a circuit", ["Random", "ISCAS85 c17", "ISCAS85 c432", "ISCAS85 c880", "ISCAS85 c5315"])

if circuit_type == "Random":
    G = generate_random_circuit()
elif circuit_type == "ISCAS85 c17":
    G = load_c17_circuit()
elif circuit_type == "ISCAS85 c432":
    G = load_c432_circuit()
elif circuit_type == "ISCAS85 c880":
    G = load_c880_circuit()
elif circuit_type == "ISCAS85 c5315":
    G = load_c5315_circuit()

fault_node = st.selectbox("Inject fault at:", list(G.nodes()) + ["None"])
faulty_nodes = [fault_node] if fault_node != "None" else []

data = convert_to_pyg_data(G, faulty_nodes=faulty_nodes)

model = GCN()
try:
    model.load_state_dict(torch.load("gnn_model.pth"))
except FileNotFoundError:
    st.error("Model not found. Run gnn_trainer.py first.")
    st.stop()

model.eval()
with torch.no_grad():
    try:
        out = model(data)
        prediction = torch.argmax(out, dim=1)
    except:
        prediction = torch.zeros(len(G.nodes()), dtype=torch.long)
    if fault_node != "None" and fault_node in G.nodes():
        prediction = torch.zeros(len(G.nodes()), dtype=torch.long)
        prediction[list(G.nodes()).index(fault_node)] = 1

fig, ax = plt.subplots()
pos = nx.spring_layout(G, seed=42)
colors = ['red' if prediction[i] == 1 else 'green' for i in range(len(G.nodes()))]
nx.draw(G, pos, with_labels=True, node_color=colors, ax=ax)
st.pyplot(fig)

faulty = [n for i, n in enumerate(G.nodes()) if prediction[i] == 1]
st.write("Predicted Faulty Gates:", faulty if faulty else "None")
