import csv
import os
from datetime import datetime

CSV_FILE = "resultado.csv"

def crear_si_no_existe():
    """Crea el archivo CSV si no existe."""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Encabezado para los parámetros
            #writer.writerow(["optimizador", "epochs", "step size", "alpha", "batch size"])
            # Encabezado para la tabla de pérdidas por época (en blanco debajo de los parámetros)
            writer.writerow([])

def guardar_parametros(opt, epochs, step_size, alpha, batch_size):
    """Agrega una fila con los parámetros usados para el entrenamiento."""
    crear_si_no_existe()
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["optimizador", "epochs", "step size", "alpha", "batch size"])
        writer.writerow([ opt, epochs, step_size, alpha, batch_size])
        writer.writerow(["epoch number", "loss"])  # encabezado de la sección de pérdidas

def guardar_loss_por_epoch(log_loss):
    """Agrega las pérdidas por época debajo de los parámetros actuales."""
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for i, loss in enumerate(log_loss):
            writer.writerow([i, float(loss)])
        writer.writerow([])  # separación entre entrenamientos

def obtener_nombre_optimizador(func):
    """Devuelve el nombre corto del optimizador según la función recibida."""
    mapa = {
        'update_sgd': 'sgd',
        'update_adam': 'adam',
        'update_sgd_momentum': 'sgd momentum',
        'update_rmsprop': 'rmsprop'
    }
    return mapa.get(func.__name__, 'desconocido')
