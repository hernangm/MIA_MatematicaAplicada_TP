import jax.numpy as jnp
from jax import grad, jit
import helpers.configHelpers as configHelpers
from neuralNetwork import loss

def adam(params, x, y, aux, t, learningRate, config):
    beta1, beta2, delta, s, r = configHelpers.extract(config, ["beta1", "beta2", "delta", "s", "r"])
    return update_adam(params, x, y, aux, t, learningRate, beta1, beta2, delta, s, r)

@jit
def update_adam(params, x, y, t, learningRate, beta1, beta2, delta, s, r):
    # alpha = 0.01
    # beta1 = 0.9
    # beta2 = 0.999
    # delta = 1e-8 #10**-8

    gradient = grad(loss)(params, x, y)

    s = s * beta1 + (1 - beta1) * gradient
    r = r * beta2 + (1 - beta2) * jnp.square(gradient)

    s_hat = s / (1 - (beta1 ** t))
    r_hat = r / (1 - (beta2 ** t))

    params = params - learningRate/(jnp.sqrt(r_hat) + delta) * s_hat
    return params, r, s
