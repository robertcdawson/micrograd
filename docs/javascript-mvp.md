# JavaScript MVP Reimplementation

This folder provides a dependency-free JavaScript recreation of the core micrograd engine.

## Files

- `js/engine.js`: `Value` class with autodiff and scalar ops.
- `js/nn.js`: `Module`, `Neuron`, `Layer`, and `MLP`.
- `js/test_engine.js`: test coverage for representative forward/backward behavior.
- `js/train_xor.js`: simple XOR-style training example.

## Runtime

- Requires only **Node.js** (no npm dependencies).
- Uses CommonJS `require` for simplicity.

## Quickstart

```bash
node js/test_engine.js
node js/train_xor.js
```

## Runtime requirement

- You only need Node.js to run the JavaScript engine, tests, and training demo.

## Design choices for MVP

1. **Scalar-only graph**
   - Mirrors the original micrograd design: each scalar operation makes one `Value` node.
   - Vector/matrix math is intentionally represented as many scalar operations.

2. **Reverse-mode autodiff**
   - `backward()` gathers a topological ordering of the dynamic graph.
   - Gradients are propagated in reverse topological order.

3. **Minimal neural net API**
   - Enough to express small MLPs and SGD loops.
   - No optimizer classes, no data loaders, no serialization.

## Known limitations / where it might fail

1. **No graph lifecycle management**
   - Large or long-running training loops can accumulate temporary objects and put pressure on memory if references are accidentally retained.

2. **Numerical stability is basic**
   - No clipping, epsilon tricks, or stabilized losses.
   - Large magnitudes can produce `Infinity`, `-Infinity`, or `NaN`.

3. **ReLU derivative at 0**
   - Uses a hard 0/1 branch and returns gradient 0 at exactly 0 (a common but not universal choice).

4. **Limited operator surface**
   - Implements core arithmetic + power + ReLU only.
   - No transcendental functions (`exp`, `log`, etc.) in this MVP.

5. **No deterministic seeding utility**
   - Weight initialization uses `Math.random()`, so runs vary.

6. **No broadcasting or tensor semantics**
   - Inputs must match expected dimensions manually.

If needed, the safest next increment is to add deterministic RNG and a few numerically stable loss helpers.
