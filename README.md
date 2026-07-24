# grokking-mechanism

Minimal notes for **understanding grokking** with a **jax-js** style workflow.

## What is grokking?
Grokking is a training pattern where a model first memorizes (low train loss, poor test accuracy), then much later suddenly generalizes well without changing architecture or data.

## Suggested experiment (small + reproducible)
1. Task: modular arithmetic classification (for example `(a + b) mod p`).
2. Model: tiny MLP or transformer.
3. Training: full-batch AdamW with non-zero weight decay.
4. Observe over many steps:
   - train loss
   - test loss
   - train/test accuracy
   - parameter norms

## Why this setup shows grokking
- Modular arithmetic is easy to memorize but has a clean algebraic rule.
- With enough training + regularization, optimization can transition from memorized lookup behavior to rule-like internal structure.

## jax-js oriented loop sketch
```js
for (let step = 0; step < maxSteps; step++) {
  const { loss, grads } = valueAndGrad(params, trainBatch);
  params = adamwUpdate(params, grads, lr, weightDecay);

  if (step % evalEvery === 0) {
    const trainAcc = accuracy(params, trainSet);
    const testAcc = accuracy(params, testSet);
    log({ step, loss, trainAcc, testAcc, norm: l2Norm(params) });
  }
}
```

A grokking run typically looks like:
- early: `trainAcc ≈ 1.0`, `testAcc` low
- late: sharp `testAcc` jump toward high generalization

## Practical tips
- Keep dataset tiny enough to memorize.
- Train much longer than usual.
- Sweep `weightDecay` and random seed.
- Save metrics each evaluation step so the delayed transition is visible.