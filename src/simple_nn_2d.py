import os
import sys
import jax.numpy as jnp
from jax import grad, jit, vmap, random, nn
import matplotlib.pyplot as plt

from backend.utils_api import enviar_loss, enviar_prediccion
from logger_csv import guardar_parametros, guardar_loss_por_epoch, obtener_nombre_optimizador
from nn_functions import (
    init_network_params, pack_params, layer_sizes,
    update_rmsprop, update_sgd, update_adam, update_sgd_momentum,
    get_batches, loss, batched_predict, pow_schedule
)

# ===========================
# 🔹 INICIALIZACIÓN DE DATOS
# ===========================

print("Inicializando datos...")

# Cargar campo desde archivo .npy
dir_path = os.path.dirname(os.path.realpath(__file__))
field_path = os.path.join(dir_path, 'field.npy')
field = jnp.load(field_path)
field = (field - field.mean()) / field.std()
field = jnp.array(field, dtype=jnp.float32)

# Dimensiones
nx, ny = field.shape

# Coordenadas normalizadas en el rango [-1, 1]
xx = jnp.linspace(-1, 1, nx)
yy = jnp.linspace(-1, 1, ny)
xx, yy = jnp.meshgrid(xx, yy, indexing='ij')
xx = jnp.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)

# Valores objetivo
ff = field.reshape(-1, 1)

# ================================
# 🔹 CONFIGURACIÓN DEL ENTRENAMIENTO
# ================================

# Ejemplo por defecto (puede sobrescribirse desde main.py)
DEFAULT_OPTIMIZER = update_rmsprop
DEFAULT_EPOCHS = 10
DEFAULT_STEP_SIZE = 0.001
DEFAULT_ALPHA = 0.02
DEFAULT_BATCH_SIZE = 32
# Variable global para detener entrenamiento
stop_flag = {"value": False}

# ===========================
# 🔹 FUNCIÓN DE ENTRENAMIENTO
# ===========================
def entrenar_modelo(update, num_epochs=10, step_size=0.001, alpha_inicial=0.02, batch_size=32, r=0.0, s=0.0, beta=0.01):

    """Entrena la red neuronal usando el optimizador y parámetros especificados."""

    # Inicializar parámetros del modelo
    key = random.key(0)
    params = init_network_params(layer_sizes, key)
    params = pack_params(params)

    # Inicializar variable auxiliar para optimizadores que lo usan
    xi, yi = next(get_batches(xx, ff, bs=batch_size))
    grads = grad(loss)(params, xi, yi)
    aux = jnp.square(grads)
    momentum = 0.0
    # Variables para Adam
    #r = 0.0
    #s = 0.0

    log_train = []

    for epoch in range(num_epochs):
        if stop_flag["value"]:
            print("Entrenamiento interrumpido por el usuario.")
            break
        idxs = random.permutation(random.key(epoch), xx.shape[0])
        alpha = alpha_inicial
        iteration = 1

        for xi, yi in get_batches(xx[idxs], ff[idxs], bs=batch_size):
            if update == update_adam:
                params, r, s = update(params, xi, yi, r, s, iteration, alpha)
            elif update == update_sgd_momentum:
                params, momentum = update(params, xi, yi, step_size, momentum, beta)
            else:
                params, aux = update(params, xi, yi, step_size, aux)

            iteration += 1
            alpha = pow_schedule(alpha, iteration)


        train_loss = loss(params, xx, ff)
        log_train.append(train_loss)
        enviar_loss(train_loss, epoch)
        pred = batched_predict(params, xx).reshape((nx, ny)).T
        enviar_prediccion(pred, epoch)
        print(f"Epoch {epoch}, Loss: {train_loss}")
    guardar_parametros(opt=obtener_nombre_optimizador(update), epochs=num_epochs, step_size=step_size, alpha=alpha_inicial, batch_size=batch_size)
    guardar_loss_por_epoch(log_train)
    ##Plot loss function
    # plt.figure()
    # plt.semilogy(log_train)
    ## Plot results
    # plt.figure()
    # plt.imshow(ff.reshape((nx, ny)).T, origin='lower', cmap='jet')
    #
    # plt.figure()
    # plt.imshow(batched_predict(params, xx).reshape((nx, ny)).T, origin='lower', cmap='jet')
    #
    ## Show figures
    # plt.show()

    return params, log_train



