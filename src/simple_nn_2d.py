import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import nn

import matplotlib.pyplot as plt

from nn_functions import init_network_params, pack_params, layer_sizes
from nn_functions import update_rmsprop, update_sgd
from nn_functions import get_batches, loss, batched_predict

# Load data
field = jnp.load('field.npy')
field = field - field.mean()
field = field / field.std()
field = jnp.array(field, dtype=jnp.float32)
nx, ny = field.shape
xx = jnp.linspace(-1, 1, nx)
yy = jnp.linspace(-1, 1, ny)
xx, yy = jnp.meshgrid(xx, yy, indexing='ij')
xx = jnp.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
ff = field.reshape(-1, 1)

# Parameters
num_epochs = 10
params = init_network_params(layer_sizes, random.key(0))
params = pack_params(params)

# optimizer
update = update_rmsprop
# update = update_sgd
step_size = 0.001

# initialize gradients
xi, yi = next(get_batches(xx, ff, bs=32))
grads = grad(loss)(params, xi, yi)
aux = jnp.square(grads)

# Training
log_train = []
for epoch in range(num_epochs):
    # Update on each batch
    idxs = random.permutation(random.key(0), xx.shape[0])
    for xi, yi in get_batches(xx[idxs], ff[idxs], bs=32):
        params, aux = update(params, xi, yi, step_size, aux)

    train_loss = loss(params, xx, ff)
    log_train.append(train_loss)
    print(f"Epoch {epoch}, Loss: {train_loss}")

# Plot loss function
plt.figure()
plt.semilogy(log_train)

# Plot results
plt.figure()
plt.imshow(ff.reshape((nx, ny)).T, origin='lower', cmap='jet')

plt.figure()
plt.imshow(batched_predict(params, xx).reshape((nx, ny)).T, origin='lower', cmap='jet')

# Show figures
plt.show()
