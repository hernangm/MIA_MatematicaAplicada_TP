import jax.numpy as jnp
from jax import grad
from jax import random
import numpy as np

import matplotlib.pyplot as plt

from nn_functions import init_network_params, pack_params, layer_sizes
from nn_functions import update_rmsprop, update_sgd, update_adam
from nn_functions import get_batches, loss, batched_predict, pow_schedule, batched_activations, hessian_eigenvals

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
num_epochs = 200
params = init_network_params(layer_sizes, random.key(0))
params = pack_params(params)

# optimizer
#update = update_rmsprop
# update = update_sgd
update = update_adam
step_size = 0.001

# initialize gradients
xi, yi = next(get_batches(xx, ff, bs=64))
grads = grad(loss)(params, xi, yi)
aux = jnp.square(grads)
eigValsArray = None
alpha = 0.04
alphas = []
epochs = []
alphas.append(alpha)
epochs.append(1)
act_matrix = [jnp.array([]), jnp.array([])]
# Training
log_train = []
for epoch in range(num_epochs):
    # Update on each batch
    idxs = random.permutation(random.key(0), xx.shape[0])
    r = 0
    s = 0
    iteration = 1
    beta1 = 0.9
    beta2 = 0.999
    delta = 1e-3 #10**-8
    schedule_r = 450
    
    for xi, yi in get_batches(xx[idxs], ff[idxs], bs=64):
        #params, aux = update(params, xi, yi, step_size, aux)
        '''Adam algorithm'''
        params, r, s = update(params, xi, yi, r, s, iteration, alpha, beta1, beta2, delta)
        iteration += 1
    
    #eigvals = hessian_eigenvals(params, xx, ff, k=2)
    #eigValsArray = jnp.vstack([eigValsArray, eigvals]) if epoch > 0 else eigvals[None, :]
    train_loss = loss(params, xx, ff)
    acts = batched_activations(params, xx)
    for i, activation in enumerate(acts):
        flat = jnp.ravel(activation)
        act_matrix[i] = jnp.concatenate([act_matrix[i], flat])
    
    print(f"Epoch {epoch}, Loss: {train_loss}")
    if len(log_train) > 1 and train_loss > log_train[-1]:
        alpha = pow_schedule(alpha, epoch + 1, schedule_r)
        alphas.append(alpha.item())
        epochs.append(epoch + 1)
        print("cambio alfa")
    log_train.append(train_loss)

# Plot loss function
plt.figure()
plt.title('Loss ADAM. scheduler power 75 epochs')
plt.xlabel('Epochs')
plt.ylabel('Costo f(x)')
plt.semilogy(log_train)

#activation histogram
plt.figure()
for i, layer_values in enumerate(act_matrix):
    flat = jnp.ravel(layer_values)
    plt.hist(flat, bins=30, alpha=0.5, label=f'activation layer {i}')
    plt.legend()
plt.title('Activation Histogram')

#alpha variation with schedule
plt.figure()
plt.title('Alpha variation with power schedule 75 epochs')
plt.plot(epochs, alphas, drawstyle='steps-pre')

# Plot results
plt.figure()
plt.imshow(ff.reshape((nx, ny)).T, origin='lower', cmap='jet')

plt.figure()
plt.imshow(batched_predict(params, xx).reshape((nx, ny)).T, origin='lower', cmap='jet')

# Show figures
plt.show()

