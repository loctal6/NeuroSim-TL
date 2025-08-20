# NeuroSim-TL: A Neuro-Symbolic Framework for Fault Localization in VLSI Circuits

## Introduction

Fault localization in VLSI (Very Large Scale Integration) circuits is a critical step in testing and verification. Traditionally, symbolic logic techniques such as SAT (Boolean Satisfiability) and SMT (Satisfiability Modulo Theory) solvers, along with classical Machine Learning methods like Decision Trees (DT) and Support Vector Machines (SVM), have been used.

However, these approaches:

* Depend heavily on carefully designed test vectors.
* Do not generalize well to new or unseen circuits.
* Become computationally expensive as circuit size grows.

This project introduces a **Graph Neural Network (GNN)**-based methodology for fault localization, capable of identifying faulty nodes in circuits by learning directly from circuit structure and gate dependencies. Our framework, **NeuroSim-TL**, demonstrates improved accuracy and scalability compared to classical baselines and provides an intuitive GUI for visualization.

---

## One-Shot Fault Localization

A central contribution of this project is the ability to achieve **one-shot fault localization**.

* **Traditional requirement:** Classical techniques require multiple test vectors to excite faults and observe faulty behavior across circuit outputs. Each fault scenario must be simulated and compared exhaustively.
* **Our approach:** By representing the circuit as a graph and using a trained GNN, the model can infer fault locations in a single forward pass, without needing multiple test patterns.
* **Why it works:** The GNN learns structural dependencies and relational patterns among gates. This allows it to detect anomalies (faulty nodes) directly from the circuit’s graph embedding.
* **Advantage:** This enables **zero-test-vector fault localization**, significantly reducing test time and making the approach more scalable for large designs.

In summary, one-shot fault localization means the model can localize faults across an entire circuit graph in one inference step, rather than relying on repeated test simulations.

---

## Methodology

1. **Circuit Representation**

   * ISCAS’85 benchmark circuits (C17, C432, C880, C5315) are parsed into graph structures.
   * Nodes represent logic gates, edges represent interconnections, and node features represent gate types using one-hot encoding.

2. **Model**

   * A three-layer Graph Convolutional Network (GCN) processes the graph.
   * The model performs node classification, predicting whether each gate is faulty or not.

3. **Baselines**

   * Decision Trees (DT)
   * Support Vector Machines (SVM)

4. **Comparison**

   * Accuracy metrics are used to compare methods. Example:

     ```
     Accuracy:
       GNN  = 0.85
       DT   = 0.72
       SVM  = 0.69
     ```

5. **Visualization GUI (Streamlit)**

   * Interactive tool to inject faults into different nodes.
   * The trained GNN highlights predicted faulty nodes in red.
   * Demonstrates one-shot localization in real time.

---

## Project Structure

```
├── model.py              # Defines the GNN model (3-layer GCN)
├── data_generator.py     # Generates circuit graphs (C17, C432, C880, C5315)
├── gnn_trainer.py        # Trains the GNN and saves model weights
├── compare_baselines.py  # Runs Decision Tree & SVM for comparison
├── streamlit_app.py      # GUI for visualization and fault injection
└── README.md             # Documentation
```

---

## Installation

```bash
# Clone repo
git clone https://github.com/your-repo/neurosim-tl.git
cd neurosim-tl

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Train the GNN

```bash
python gnn_trainer.py
```

* Trains the GNN and saves weights as `gnn_model.pth`.

### Compare with Baselines

```bash
python compare_baselines.py
```

* Prints accuracy of GNN, DT, and SVM.

### Launch GUI

```bash
streamlit run streamlit_app.py
```

* Opens the visualization dashboard in the browser.
* Select a circuit, inject a fault, and observe predicted faulty nodes.

---

## Results

* The GNN achieves higher accuracy than Decision Trees and SVM.
* Fault localization is performed in a **one-shot manner**, without test vectors.
* The GUI demonstrates interactive and visual fault injection analysis.

---

## Future Scope

* Extend to larger ISCAS circuits such as C7552.
* Explore transfer learning for cross-circuit generalization.
* Combine GNNs with symbolic SAT/SMT methods for hybrid neuro-symbolic reasoning.
* Integration into EDA tools for real-time fault diagnosis and testing.

---

## Authors

* Ashutosh Singh (IEC2022053)
* Project under the guidance of Dr. Sunny Sharma, Assistant Professor, IIIT Allahabad

---
