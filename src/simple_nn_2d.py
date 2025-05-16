import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import nn
import sys

import matplotlib.pyplot as plt

from nn_functions import init_network_params, pack_params, layer_sizes
from nn_functions import update_rmsprop, update_sgd, update_adam,update_sgd_momentum
from nn_functions import get_batches, loss, batched_predict, pow_schedule


###############################################################################
###############################################################################

# Argumentos
opt = int(sys.argv[1]) if len(sys.argv) > 1 else 1
num_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
step_size = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
alpha_init = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02

# Elegir optimizador
if opt == 1:
    print("Usando optimizador: SGD")
    update = update_sgd
elif opt == 2:
    print("Usando optimizador: RMSProp")
    update = update_rmsprop
elif opt == 3:
    print("Usando optimizador: Adam")
    update = update_adam
elif opt == 4:
    print("Usando optimizador: SGD con Momentum")
    update = update_sgd_momentum
else:
    raise ValueError("Elegí una opción válida: 1, 2, 3 o 4.")

print(f"Entrenando por {num_epochs} épocas con step_size={step_size}, alpha={alpha_init}...")

###############################################################################
###############################################################################

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
#update = update_rmsprop
#update = update_sgd
#update = update_adam
step_size = 0.0599999       

# initialize gradients
xi, yi = next(get_batches(xx, ff, bs=32))
grads = grad(loss)(params, xi, yi)
aux = jnp.square(grads)

# Training
log_train = []
for epoch in range(num_epochs):
    # Update on each batch
    idxs = random.permutation(random.key(0), xx.shape[0])
    r = 0
    s = 0
    iteration = 1
    alpha = 0.02
    for xi, yi in get_batches(xx[idxs], ff[idxs], bs=32):
        if opt in [1, 2]:
            params, aux = update(params, xi, yi, step_size, aux)
        elif opt == 3:
            params, r, s = update(params, xi, yi, r, s, iteration, alpha)
        #params, aux = update(params, xi, yi, step_size, aux)
        '''Adam algorithm'''
        #params, r, s = update(params, xi, yi, r, s, iteration, alpha)
        iteration += 1
        alpha = pow_schedule(alpha, iteration)

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
