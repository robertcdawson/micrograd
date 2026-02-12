const { Value } = require('./engine');
const { MLP } = require('./nn');

const samples = [
  { x: [0, 0], y: 0 },
  { x: [0, 1], y: 1 },
  { x: [1, 0], y: 1 },
  { x: [1, 1], y: 0 },
];

const model = new MLP(2, [4, 4, 1]);
const learningRate = 0.05;
const steps = 300;

for (let step = 0; step < steps; step += 1) {
  let totalLoss = new Value(0);

  for (const sample of samples) {
    const pred = model.forward(sample.x);
    const target = new Value(sample.y);
    const loss = pred.sub(target).pow(2);
    totalLoss = totalLoss.add(loss);
  }

  model.zeroGrad();
  totalLoss.backward();

  for (const p of model.parameters()) {
    p.data += -learningRate * p.grad;
  }

  if ((step + 1) % 50 === 0) {
    // eslint-disable-next-line no-console
    console.log(`step=${step + 1}, loss=${totalLoss.data.toFixed(4)}`);
  }
}

console.log('\nPredictions after training:');
for (const sample of samples) {
  const pred = model.forward(sample.x);
  console.log(`${sample.x.join(', ')} -> ${pred.data.toFixed(3)} (target=${sample.y})`);
}
