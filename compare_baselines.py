import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from model import GCN
from data_generator import generate_random_circuit, convert_to_pyg_data

# Generate dataset
X, Y = [], []
for _ in range(100):
    G = generate_random_circuit()
    fault_node = [random.choice(list(G.nodes()))] if random.random() < 0.3 else []
    data = convert_to_pyg_data(G, faulty_nodes=fault_node)
    x_mean = data.x.mean(dim=0).tolist()
    label = 1 if fault_node else 0
    X.append(x_mean)
    Y.append(label)

# Train and evaluate baselines
dt = DecisionTreeClassifier().fit(X, Y)
svm = SVC().fit(X, Y)

print("Decision Tree Accuracy:", dt.score(X, Y))
print("SVM Accuracy:", svm.score(X, Y))
