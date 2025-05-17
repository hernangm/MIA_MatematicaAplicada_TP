# utils_api.py

import requests

def enviar_loss(valor_loss, epoch=None, url="http://localhost:8000/loss"):
    """Envía un valor de pérdida a la API en tiempo real"""
    try:
        payload = {"loss": float(valor_loss)}
        requests.post(url, json=payload)
    except Exception as e:
        print("Error al enviar loss:", e)


def enviar_prediccion(pred_array, epoch=None, url="http://localhost:8000/pred"):
    try:
        payload = {
            "pred": pred_array.tolist(),  # convertir jax array a lista
            "epoch": epoch
        }
        requests.post(url, json=payload)
    except Exception as e:
        print("Error al enviar predicción:", e)
