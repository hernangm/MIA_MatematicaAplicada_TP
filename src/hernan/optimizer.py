import jax.numpy as jnp
from jax import random, grad
import helpers.configHelpers as configHelpers
from neuralNetwork import batched_predict, loss

def _getBatches(x, y, bs):
    for i in range(0, len(x), bs):
        yield x[i:i+bs], y[i:i+bs]


def optimize(params, data, config, algorithm, algorithmConfig, scheduler, schedulerConfig):
    trainLog = []
    epochs, minibatch = configHelpers.extract(config, ["epochs", "minibatch"])
    learningRate = configHelpers.extract(algorithmConfig, ["learningRate"])
    xx, ff = data
    xi, yi = next(_getBatches(xx, ff, bs=minibatch))
    grads = grad(loss)(params, xi, yi)
    aux = jnp.square(grads)
    for epoch in range(epochs):
        idxs = random.permutation(random.key(0), xx.shape[0])
        t = 0
        for xi, yi in _getBatches(xx[idxs], ff[idxs], bs=minibatch):
            t += 1
            params, aux = algorithm(params, xi, yi, aux, t, learningRate, algorithmConfig)

        learningRate = scheduler(learningRate, t, schedulerConfig)
        trainLoss = loss(params, xx, ff)
        trainLog.append({
            "epoch": epoch,
            "trainLoss": trainLoss,
            "learningRate": learningRate
        })
    return (params, batched_predict(params, xx), trainLog)
