from jax import grad, jit
from neuralNetwork import loss

def sgd(params, x, y, aux, t, learningRate):
    return update_sgd(params, x, y, aux, learningRate)

@jit
def update_sgd(params, x, y, aux, learningRate):
    grads  = grad(loss)(params, x, y)
    params = params - learningRate * grads
    return params, aux