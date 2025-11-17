# Transformer & RNN Mini Assignments  
**Course:** Advanced Neural Networks  
**Student:** Aravind Reddy  
**Student ID:** 700761877
**Semester:** Fall 2025  


This repository contains my implementation of three core deep learning components covered in lecture:  
1. Character-Level RNN Language Model  
2. Mini Transformer Encoder  
3. Scaled Dot-Product Attention  

The goal of this repository is to demonstrate understanding of sequence modeling, attention mechanisms, and Transformer building blocks using PyTorch in Google Colab.

---

# 📌 **Q1 — Character-Level RNN Language Model**

### **Overview**
This experiment builds a character-level language model that predicts the next character given previous characters.  
It uses:
- Embedding layer  
- GRU (can also use Vanilla RNN or LSTM)  
- Linear output layer  
- Temperature-based sampling  
- Teacher forcing during training  

### **Features**
- Train on a toy dataset + 50–200 KB text (e.g., Sherlock Holmes dataset)
- Sequence length 50–100  
- Hidden size 64–256  
- Adam optimizer  
- Cross-entropy loss  
- Temperature-controlled sampling for text generation  

### **Outputs Included**
- Training loss curve  
- Sampled text for temperatures τ = 0.7, 1.0, 1.2  
- Reflection paragraph on model hyperparameters  

---

# 📌 **Q2 — Mini Transformer Encoder**

### **Overview**
This part implements a **single-layer Transformer Encoder** from scratch (no PyTorch `nn.Transformer`), covering:

### **Components Implemented**
- Tokenization + Vocabulary building  
- Embedding  
- Sinusoidal positional encoding  
- Multi-head self-attention (4 heads)  
- Scaled dot-product attention  
- Add & Norm  
- Feed-forward network  
- Contextual embeddings output  

### **Outputs Included**
- Input token IDs  
- Final contextual embeddings  
- Word-to-word attention heatmap  

### **Purpose**
This demonstrates understanding of:
- Query/Key/Value operations  
- Multi-head splitting and merging  
- Transformer computation flow  
- Position encoding behavior  

---

# 📌 **Q3 — Scaled Dot-Product Attention**

### **Overview**
This part implements the **core attention formula** from the slides:

\[
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V
\]

### **Delivered Requirements**
- A PyTorch implementation of the attention function  
- Random test Q, K, V tensors  
- Printed outputs:
  - Raw QKᵀ scores  
  - Scaled scores  
  - Softmax before vs after scaling (numerical stability check)  
  - Attention weight matrix  
  - Final attention output vectors  

### **Purpose**
To show how scaling prevents softmax saturation and improves gradient stability.

---

# 📂 **Repository Structure**
