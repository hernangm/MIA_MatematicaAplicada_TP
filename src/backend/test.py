import requests
import time
import numpy as np

for i in range(50):
    loss_value = float(np.exp(-i * 0.1))  # Decreciente
    print(f"Enviando loss: {loss_value:.5f}")
    try:
        requests.post("http://localhost:8000/loss", json={"loss": loss_value})
    except Exception as e:
        print("Error:", e)
    time.sleep(0.2)