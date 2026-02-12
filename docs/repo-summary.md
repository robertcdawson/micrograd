# Repository Summary

## What this repository is
`micrograd` is a compact educational implementation of reverse-mode automatic differentiation for scalar values, plus a tiny neural network API built on top.

## Core components

- **Autograd engine (`micrograd/engine.py`)**
  - `Value` holds:
    - a scalar `data`
    - a scalar gradient `grad`
    - links to predecessor nodes (`_prev`)
    - an operation tag (`_op`) for debugging/visualization
  - Supports arithmetic (`+`, `-`, `*`, `/`, powers), unary negation, and `relu`.
  - Builds a dynamic DAG during forward execution.
  - Executes `backward()` by:
    1. topologically sorting the graph
    2. visiting nodes in reverse topological order to apply local derivatives

- **Neural net module (`micrograd/nn.py`)**
  - `Module` base class with `parameters()` and `zero_grad()`.
  - `Neuron`: weighted sum + bias + optional ReLU.
  - `Layer`: list of neurons.
  - `MLP`: stack of layers, final layer linear by default.

- **Tests (`test/test_engine.py`)**
  - Compare outputs and gradients against a trusted reference implementation for parity.
  - Cover both simple and composite operator chains.

## Why it matters
Despite its tiny size, the code demonstrates the same conceptual core used by larger deep learning frameworks: construct a computation graph on the forward pass, then run reverse-mode autodiff to obtain gradients.
