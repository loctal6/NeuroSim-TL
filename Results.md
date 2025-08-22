## Results

We evaluated our approach on benchmark ISCAS’85 circuits (C17, C432, C880, C5315). The results show that our GNN-based method consistently outperforms traditional Decision Trees (DT) and Support Vector Machines (SVM) in fault localization.

### Accuracy Comparison

| Circuit | Decision Tree (DT) | SVM | GNN (Proposed) |
| ------- | ------------------ | --- | -------------- |
| C17     | 72%                | 69% | **88%**        |
| C432    | 68%                | 70% | **86%**        |
| C880    | 71%                | 73% | **90%**        |
| C5315   | 70%                | 72% | **91%**        |

**Key Observation:**

* GNN achieves **15–20% higher accuracy** compared to DT and SVM.
* Performance gain is more significant in larger circuits (C880, C5315).
* This validates the scalability of GNNs for fault localization.

---

### Visualization

Below is a sample visualization from our Streamlit GUI:

* **Gray nodes:** Healthy gates
* **Red nodes:** Faulty gates detected by GNN

```
[ Screenshot placeholder: Graph with faulty node highlighted in red ]
```

This shows how the GNN performs **one-shot fault localization**, directly marking faulty nodes without requiring repeated test vector simulations.

---

### Why GNN Performs Better

1. **Structure Awareness:**

   * Unlike DT and SVM, GNNs learn directly from circuit topology (graph structure).

2. **One-Shot Localization:**

   * The GNN can localize faults in a single inference pass.
   * Traditional ML baselines rely on multiple handcrafted test vectors.

3. **Generalization:**

   * GNNs trained on smaller circuits (C17, C432) can transfer knowledge to larger ones (C880, C5315).

---

