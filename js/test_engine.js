const assert = require('node:assert/strict');
const { Value } = require('./engine');

const close = (left, right, tol = 1e-6) => Math.abs(left - right) < tol;

function testMoreOpsParity() {
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

  assert.equal(close(g.data, 24.70408163265306), true, 'forward pass should match reference output');

  g.backward();

  assert.equal(close(a.grad, 138.83381924198252), true, 'gradient for a should match reference');
  assert.equal(close(b.grad, 645.5772594752186), true, 'gradient for b should match reference');
}

function testSanity() {
  const x = new Value(-4.0);
  const z = x.mul(2).add(2).add(x);
  const q = z.relu().add(z.mul(x));
  const h = z.mul(z).relu();
  const y = h.add(q).add(q.mul(x));

  y.backward();

  assert.equal(y.data, -20);
  assert.equal(x.grad, 46);
}

function run() {
  testSanity();
  testMoreOpsParity();
  // eslint-disable-next-line no-console
  console.log('All JavaScript engine tests passed.');
}

if (require.main === module) {
  run();
}

module.exports = {
  run,
};
