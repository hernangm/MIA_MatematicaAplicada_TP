import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import nn

import matplotlib.pyplot as plt

def pack_params(params):
    """Pack parameters into a single vector."""
    return jnp.concatenate([jnp.ravel(w) for w, _ in params] +
                             [jnp.ravel(b) for _, b in params])

layer_sizes = [2, 64, 64, 1]
def unpack_params(params):
    """Unpack parameters from a single vector."""
    weights = []
    for i in range(len(layer_sizes) - 1):
        weight_size = layer_sizes[i] * layer_sizes[i + 1]
        to_unpack, params = params[:weight_size], params[weight_size:]
        weights.append(jnp.array(to_unpack).reshape(layer_sizes[i + 1], layer_sizes[i]))

    biases = []
    for i in range(len(layer_sizes) - 1):
        bias_size = layer_sizes[i + 1]
        to_unpack, params = params[:bias_size], params[bias_size:]
        biases.append(jnp.array(to_unpack).reshape(layer_sizes[i + 1]))

    params = [(w, b) for w, b in zip(weights, biases)]
    return params

def random_layer_params(m, n, key, scale=1e-2):
    ''' Randomly initialize weights and biases for a dense neural network layer '''
    w_key, b_key = random.split(key)
    scale = jnp.sqrt(6.0 / (m + n))
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))
    # return jnp.ones((n, m)), jnp.zeros((n,))

def init_network_params(sizes, key):
    ''' Initialize all layers for a fully-connected neural network with sizes "sizes" '''
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

@jit
def predict(params, coord):
    params = unpack_params(params)
    hidden = coord
    for w, b in params[:-1]:
        outputs = jnp.dot(w, hidden) + b
        hidden = nn.tanh(outputs)

    final_w, final_b = params[-1]
    output = jnp.dot(final_w, hidden) + final_b
    return output
batched_predict = vmap(predict, in_axes=(None, 0))

def loss(params, coord, target):
    preds = batched_predict(params, coord)
    return jnp.mean(jnp.square(preds - target))

@jit
def update_sgd(params, x, y, step, aux):
    grads  = grad(loss)(params, x, y)
    params = params - step * grads
    return params, aux

@jit
def update_rmsprop(params, x, y, step_size, aux):
    beta = 0.9
    grads = grad(loss)(params, x, y)
    aux = beta * aux + (1 - beta) * jnp.square(grads)
    step_size = step_size / (jnp.sqrt(aux) + 1e-8)
    params = params - step_size * grads
    return params, aux

def get_batches(x, y, bs):
    for i in range(0, len(x), bs):
        yield x[i:i+bs], y[i:i+bs]
