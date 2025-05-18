import jax.numpy as jnp
from jax import grad, jit
import helpers.configHelpers as configHelpers
from neuralNetwork import loss

def rmsprop(params, x, y, aux, t, learningRate, config):
    beta = configHelpers.extract(config, ["beta"])
    return update_rmsprop(params, x, y, aux, learningRate, beta)

@jit
def update_rmsprop(params, x, y, aux, learningRate, beta):
    grads = grad(loss)(params, x, y)
    aux = beta * aux + (1 - beta) * jnp.square(grads)
    params = params - learningRate * grads
    # learningRate = learningRate / (jnp.sqrt(aux) + 1e-8)
    return params, aux