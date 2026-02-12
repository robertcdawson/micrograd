const { Value } = require('./engine');

class Module {
  zeroGrad() {
    for (const parameter of this.parameters()) {
      parameter.grad = 0;
    }
  }

  parameters() {
    return [];
  }
}

class Neuron extends Module {
  constructor(nin, nonlin = true) {
    super();
    this.w = Array.from({ length: nin }, () => new Value((Math.random() * 2) - 1));
    this.b = new Value(0);
    this.nonlin = nonlin;
  }

  forward(x) {
    if (!Array.isArray(x) || x.length !== this.w.length) {
      throw new Error(`Expected ${this.w.length} inputs, received ${Array.isArray(x) ? x.length : 'non-array input'}.`);
    }

    let act = this.b;
    for (let i = 0; i < this.w.length; i += 1) {
      act = act.add(this.w[i].mul(x[i]));
    }
    return this.nonlin ? act.relu() : act;
  }

  parameters() {
    return [...this.w, this.b];
  }

  toString() {
    return `${this.nonlin ? 'ReLU' : 'Linear'}Neuron(${this.w.length})`;
  }
}

class Layer extends Module {
  constructor(nin, nout, nonlin = true) {
    super();
    this.neurons = Array.from({ length: nout }, () => new Neuron(nin, nonlin));
  }

  forward(x) {
    const out = this.neurons.map((n) => n.forward(x));
    return out.length === 1 ? out[0] : out;
  }

  parameters() {
    return this.neurons.flatMap((n) => n.parameters());
  }

  toString() {
    return `Layer[${this.neurons.map((n) => n.toString()).join(', ')}]`;
  }
}

class MLP extends Module {
  constructor(nin, nouts) {
    super();
    const sizes = [nin, ...nouts];
    this.layers = [];

    for (let i = 0; i < nouts.length; i += 1) {
      const nonlin = i !== nouts.length - 1;
      this.layers.push(new Layer(sizes[i], sizes[i + 1], nonlin));
    }
  }

  forward(x) {
    let out = x;
    for (const layer of this.layers) {
      out = layer.forward(out);
    }
    return out;
  }

  parameters() {
    return this.layers.flatMap((layer) => layer.parameters());
  }

  toString() {
    return `MLP[${this.layers.map((layer) => layer.toString()).join(', ')}]`;
  }
}

module.exports = {
  Module,
  Neuron,
  Layer,
  MLP,
};
