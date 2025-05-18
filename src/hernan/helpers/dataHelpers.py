import jax.numpy as jnp

def loadData(filePath):
    # Load data
    field = jnp.load(filePath)
    field = field - field.mean()
    field = field / field.std()
    field = jnp.array(field, dtype=jnp.float32)
    nx, ny = field.shape
    xx = jnp.linspace(-1, 1, nx)
    yy = jnp.linspace(-1, 1, ny)
    xx, yy = jnp.meshgrid(xx, yy, indexing='ij')
    xx = jnp.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
    ff = field.reshape(-1, 1)
    return ((xx, ff), (nx, ny))