#micrograd-js

![awww](puppy.jpg)

A tiny JavaScript autograd engine for scalar values with reverse-mode autodiff and a minimal neural-network stack (`Module`, `Neuron`, `Layer`, `MLP`) built on top.

Thanks to Andrej Karpathy for the original code and project: <https://github.com/karpathy/micrograd>.

## What's in this repo

- `js/engine.js`: `Value` class, scalar ops, and backpropagation
- `js/nn.js`: `Module`, `Neuron`, `Layer`, and `MLP`
- `js/test_engine.js`: forward/backward checks for the JS engine
- `js/train_xor.js`: XOR training demo
- `docs/repo-summary.md`: repository overview
- `docs/javascript-mvp.md`: JavaScript MVP notes and limitations

## Requirements

- Node.js (no npm dependencies)

## Example usage

```javascript
const { Value } = require('./js/engine');

const a = new Value(-4.0);
const b = new Value(2.0);
let c = a.add(b);
let d = a.mul(b).add(b.pow(3));
c = c.add(c).add(1);
c = c.add(1).add(c).add(a.neg());
d = d.add(d.mul(2)).add(b.add(a).relu());
d = d.add(d.mul(3)).add(b.sub(a).relu());
const e = c.sub(d);
const f = e.pow(2);
let g = f.div(2.0);
g = g.add(new Value(10.0).div(f));

console.log(g.data.toFixed(4));
g.backward();
console.log(a.grad.toFixed(4));
console.log(b.grad.toFixed(4));
```

## Running checks

```bash
node js/test_engine.js
node js/train_xor.js
```

## License

MIT
