# LCQNN: Linear Combination of Quantum Neural Networks

This repository contains the official PyTorch implementation of the **LCQNN** framework.

LCQNN mitigates the **Barren Plateau** problem by employing a learnable superposition of multiple quantum neural network blocks (based on Linear Combination of Unitaries), balancing **expressivity** and **trainability**.

## 📂 Scripts Description

*   **`utils.py`**: Core library containing quantum gate definitions ($R_y$, CNOT), LCQNN coefficient layers, and $k$-local/global block constructors.
*   **`mnist_gradient_compute.py`**: The main script for collecting gradient statistics across different depths ($D$) and blocks ($L$).
*   **`mnist_local_qnn.py`**: Scalable training on MNIST using the **$k$-local** architecture.
*   **`mnist_global_qnn.py`**: Training on binary MNIST tasks using the **Global** entanglement architecture.
*   **`Iris_global_qnn.py`**: A simple example on the Iris dataset.
*   **`lcqnn_final.ipynb`**: A consolidated demonstration notebook showing the complete pipeline.
*   **`grad_analyze.ipynb`**: An interactive notebook for analyzing gradient variance scaling to visualize the mitigation of Barren Plateaus.

## 🛠️ Requirements

```bash
pip install torch torchvision scikit-learn numpy matplotlib tqdm
```

## 📝 Citation


```bibtex
@article{yao2025lcqnn,
  title={LCQNN: Linear Combination of Quantum Neural Networks},
  author={Yao, Hongshun and Liu, Xia and Jing, Mingrui and Li, Guangxi and Wang, Xin},
  journal={arXiv preprint arXiv:2507.02832},
  year={2025}
}
```
