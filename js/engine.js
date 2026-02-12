class Value {
  constructor(data, children = [], op = '') {
    if (typeof data !== 'number' || Number.isNaN(data)) {
      throw new TypeError(`Value data must be a valid number. Received: ${data}`);
    }

    this.data = data;
    this.grad = 0;
    this._backward = () => {};
    this._prev = new Set(children);
    this._op = op;
  }

  static from(value) {
    return value instanceof Value ? value : new Value(value);
  }

  add(other) {
    const rhs = Value.from(other);
    const out = new Value(this.data + rhs.data, [this, rhs], '+');

    out._backward = () => {
      this.grad += 1 * out.grad;
      rhs.grad += 1 * out.grad;
    };

    return out;
  }

  sub(other) {
    const rhs = Value.from(other);
    return this.add(rhs.neg());
  }

  mul(other) {
    const rhs = Value.from(other);
    const out = new Value(this.data * rhs.data, [this, rhs], '*');

    out._backward = () => {
      this.grad += rhs.data * out.grad;
      rhs.grad += this.data * out.grad;
    };

    return out;
  }

  div(other) {
    const rhs = Value.from(other);
    return this.mul(rhs.pow(-1));
  }

  pow(power) {
    if (typeof power !== 'number' || Number.isNaN(power)) {
      throw new TypeError(`Power must be a valid number. Received: ${power}`);
    }

    const out = new Value(this.data ** power, [this], `**${power}`);

    out._backward = () => {
      this.grad += power * (this.data ** (power - 1)) * out.grad;
    };

    return out;
  }

  relu() {
    const out = new Value(this.data < 0 ? 0 : this.data, [this], 'ReLU');

    out._backward = () => {
      this.grad += (out.data > 0 ? 1 : 0) * out.grad;
    };

    return out;
  }

  neg() {
    return this.mul(-1);
  }

  backward() {
    const topo = [];
    const visited = new Set();

    const buildTopo = (value) => {
      if (!visited.has(value)) {
        visited.add(value);
        for (const child of value._prev) {
          buildTopo(child);
        }
        topo.push(value);
      }
    };

    buildTopo(this);

    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i -= 1) {
      topo[i]._backward();
    }
  }

  toString() {
    return `Value(data=${this.data}, grad=${this.grad})`;
  }
}

module.exports = {
  Value,
};
