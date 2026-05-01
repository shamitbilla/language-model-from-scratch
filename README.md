# Neural Networks: From Autograd to GPT-2

Built foundational neural network components from scratch, progressing from a scalar autograd engine to a full GPT-2 reproduction with training optimizations.

## What's built

- **Micrograd** — scalar autograd engine with dynamic DAG construction and reverse-mode autodiff
- **Language Models** — bigram, MLP, and Bengio language models from scratch
- **Batch Normalization** — activations, dead neurons, normalization techniques
- **Backpropagation** — manual backprop through complex operations from first principles
- **WaveNet** — hierarchical temporal convolution architecture
- **GPT v1** — transformer language model trained on Shakespeare
- **BPE Tokenizer** — byte pair encoding implementation
- **GPT-2 (124M)** — full reproduction with training pipeline optimizations: mixed precision (bfloat16), torch.compile with kernel fusion, flash attention, gradient accumulation — ~50% speedup over naive baseline

## Stack

Python, PyTorch, NumPy
