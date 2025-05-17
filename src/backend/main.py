from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import socketio
import uvicorn
import threading
import subprocess

from simple_nn_2d import entrenar

# Instancia FastAPI por separado
fastapi_app = FastAPI()

# Permitir conexiones desde React u otros orígenes
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # cambiar por ["http://localhost:8000"] si querés limitar
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Crear el servidor de socket.io
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")

# Unir FastAPI con Socket.IO
app = socketio.ASGIApp(sio, fastapi_app)

# Endpoint HTTP para recibir un valor de pérdida
@fastapi_app.post("/loss")
async def receive_loss(data: dict):
    loss_value = float(data.get("loss", 0.0))
    epoch = int(data.get("epoch", 0))  # ✅ extrae epoch del JSON recibido
    await sio.emit("new_loss", {"loss": loss_value, "epoch": epoch})  # ✅ lo reenvía
    return {"status": "ok"}


@fastapi_app.post("/pred")
async def recibir_pred(data: dict):
    pred = data.get("pred", [])
    epoch = data.get("epoch", 0)
    await sio.emit("new_pred", {"pred": pred, "epoch": epoch})
    return {"status": "ok"}


#    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    # Lanza el servidor ASGI (Socket.IO + FastAPI)
def run_server():
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

    # Lanza el script de entrenamiento
def run_entrenamiento():
        entrenar()

    # Si se ejecuta directamente como script
if __name__ == "__main__":
    # Hilo 1: servidor
    server_thread = threading.Thread(target=run_server)

    # Hilo 2: entrenamiento
    entrenamiento_thread = threading.Thread(target=run_entrenamiento)

    # Iniciar ambos hilos
    server_thread.start()
    entrenamiento_thread.start()

    # Esperar a que terminen (si querés)
    server_thread.join()
    entrenamiento_thread.join()