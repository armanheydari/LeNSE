# LeNSE: Learning to Navigate Subgraph Embeddings for Large-Scale Combinatorial Optimisation

> A reinforcement learning framework that scales existing combinatorial optimisation heuristics to massive graphs by learning to identify and navigate optimal subgraphs — achieving near-optimal solutions at a fraction of the runtime.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Paper](https://img.shields.io/badge/ICML%202022-paper-red.svg)](https://arxiv.org/abs/2205.10106)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.5.0-orange.svg)](https://pytorch-geometric.readthedocs.io/)

## 📌 Overview

Many real-world combinatorial optimisation (CO) problems — such as Max-Cut, Vertex Cover, and Influence Maximisation — are NP-hard and formulated over graphs. Standard heuristics produce near-optimal solutions but fail to scale: running them on graphs with millions of edges is computationally prohibitive. LeNSE addresses this by learning to prune the original graph down to a small, high-quality subgraph where the heuristic can run cheaply, without sacrificing solution quality.

The core idea is a two-stage approach: first, a Graph Neural Network encoder is trained (using InfoNCE contrastive loss) to produce a discriminative Euclidean embedding of subgraphs. Then, a reinforcement learning agent navigates the embedding space — treating it as a map — to iteratively modify any random subgraph into an optimal one. Published at ICML 2022 (PMLR 162:9622–9638), LeNSE was validated on graphs with up to **10 million edges** across three CO problems.

## ✨ Key Features

* **Problem-agnostic subgraph pruning:** LeNSE identifies a small candidate subgraph from any large input graph, then hands it off to any existing heuristic. No problem-specific RL reward engineering required — the heuristic itself provides the signal.
* **GNN encoder with InfoNCE loss:** A Graph Neural Network is trained contrastively to embed subgraphs such that high-quality subgraphs (those containing near-optimal solution nodes) cluster together in Euclidean space, enabling RL navigation.
* **RL-guided exploration (`guided_exploration.py`):** A reinforcement learning agent treats the learned embedding as a navigation map, iteratively moving from a random subgraph toward the optimal region of embedding space.
* **Three supported CO problems:** Separate implementations for Max-Cut (`Max_Cut/`), Vertex Cover (`MVC/`), and Influence Maximisation (`IM/`), each with the appropriate problem-specific heuristic.
* **Scalable data pipeline:** Efficient preprocessing from raw Stanford SNAP edge-list `.txt` files through train/test graph splitting to balanced fixed-size subgraph datasets, ready for PyTorch Geometric.
* **Latent space visualisation:** An autoencoder-based visualisation tool (`autoencoder.py`) lets you inspect the learned subgraph embedding space and verify class separation before RL training.

## 🛠️ Tech Stack

* **Core Language:** Python 3.8+
* **Frameworks & Libraries:** PyTorch Geometric 2.5.0, NetworkX 2.8.8, pytorch-metric-learning
* **Graph Data:** Stanford SNAP database (real-world graphs up to 10M edges)

## 🚀 Getting Started

### Prerequisites

Python 3.8 or higher. A GPU is recommended for GNN encoder training on large graphs.

### Installation

```bash
git clone https://github.com/armanheydari/LeNSE.git
cd LeNSE
pip install networkx==2.8.8
pip install torch_geometric==2.5.0
pip install pytorch-metric-learning
```

## 💻 Usage

The pipeline has four sequential stages. Each stage is run from within the relevant problem subdirectory (`Max_Cut/`, `MVC/`, or `IM/`).

### Stage 1 — Prepare the graph

Convert a raw SNAP edge-list `.txt` file into a NetworkX graph, then split into train and test subgraphs:

```bash
python make_graph_from_edge_list.py
python get_train_test_graphs.py
```

For Influence Maximisation only, also run:
```bash
python get_graph_sample_for_IC.py
```

### Stage 2 — Generate the dataset

Run the heuristic on the train graph and produce a balanced dataset of ranked subgraphs:

```bash
python get_scores_seeds.py --graph_name facebook --budget 50
python fixed_size_dataset.py --budget 50 --num_samples 100 --fixed_size 200
```

### Stage 3 — Train the GNN encoder and visualise

```bash
python embedding_training.py
python prepare_data.py
python autoencoder.py   # optional: visualise the embedding space
```

### Stage 4 — RL training

```bash
python guided_exploration.py --encoder_name encoder
```

## 📁 Project Structure

```
LeNSE/
├── Max_Cut/          # Max-Cut problem implementation
├── MVC/              # Minimum Vertex Cover implementation
├── IM/               # Influence Maximisation implementation
│   └── get_graph_sample_for_IC.py  # IC model preprocessing
├── requirements.txt
└── README.md
```

Each problem directory shares the same structure:
```
<problem>/
├── make_graph_from_edge_list.py   # Raw SNAP .txt → NetworkX graph
├── get_train_test_graphs.py       # Train/test graph split
├── get_scores_seeds.py            # Run heuristic, collect seed nodes
├── fixed_size_dataset.py          # Build balanced subgraph dataset
├── functions.py                   # Shared utilities (subgraph generation etc.)
├── embedding_training.py          # Train GNN encoder (InfoNCE loss)
├── prepare_data.py                # Save subgraph embedding tensors
├── autoencoder.py                 # Visualise embedding space
└── guided_exploration.py          # RL training (LeNSE agent)
```

## 📄 Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{ireland2022lense,
  title     = {LeNSE: Learning To Navigate Subgraph Embeddings for Large-Scale Combinatorial Optimisation},
  author    = {Ireland, David and Montana, Giovanni},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning},
  pages     = {9622--9638},
  year      = {2022},
  volume    = {162},
  publisher = {PMLR}
}
```

## 📄 License

See the [LICENSE](LICENSE) file for details.
