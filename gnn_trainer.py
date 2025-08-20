import torch
import random
from torch_geometric.loader import DataLoader
from model import GCN
from data_generator import generate_random_circuit, convert_to_pyg_data

dataset = []
for _ in range(200):
    G = generate_random_circuit()
    fault_node = [random.choice(list(G.nodes()))] if random.random() < 0.3 else []
    data = convert_to_pyg_data(G, faulty_nodes=fault_node)
    dataset.append(data)

loader = DataLoader(dataset, batch_size=1)

model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(100):
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "gnn_model.pth")
print("✅ GNN model saved as gnn_model.pth")
