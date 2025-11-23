# Llama 2 from Scratch

A clean, minimalistic implementation of the **Llama 2 Large Language Model by Meta** in pure PyTorch. This repository focuses on readability and understanding the core components of the model. 

---

## Model Architecture & Features 🧠

This implementation faithfully replicates the Llama 2 architecture with the following technical details, including links to the foundational research papers:

* **RMSNorm** (Root Mean Square Layer Normalization): Pre-normalization using RMSNorm for improved training stability.
    * [Root Mean Square Layer Normalization](https://www.semanticscholar.org/paper/Root-Mean-Square-Layer-Normalization-Zhang-Sennrich/10eda4521c032adabaa8e70d6569e17370b29dcd)
* **SwiGLU Activation** (Swish-Gated Linear Unit): Uses the SwiGLU activation function in the feed-forward networks (FFN).
    * [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
* **Rotary Positional Embeddings (RoPE):** Implements relative positional encodings via rotation matrices at each attention layer.
    * [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
* **Grouped-Query Attention (GQA):** Optimization that shares key-value heads across multiple query heads to reduce memory bandwidth during inference (essential for 34B/70B variants, configurable here). 
    * [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245.pdf)
* **KV Caching:** Efficient caching of Key and Value states to accelerate autoregressive decoding during inference.
