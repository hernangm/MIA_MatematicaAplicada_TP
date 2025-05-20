from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware
import socketio
import uvicorn
import threading
import subprocess
from simple_nn_2d import (
    entrenar_modelo,
    update_rmsprop,
    update_sgd,
    update_sgd_momentum,
    update_adam, stop_flag
)



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

@fastapi_app.post("/reset")
async def reset_entrenamiento():
    stop_flag["value"] = True
    return {"status": "entrenamiento_reset"}



@fastapi_app.post("/pred")
async def recibir_pred(data: dict):
    pred = data.get("pred", [])
    epoch = data.get("epoch", 0)
    await sio.emit("new_pred", {"pred": pred, "epoch": epoch})
    return {"status": "ok"}

#@fastapi_app.post("/entrenar")
#def iniciar_entrenamiento():
#    thread = threading.Thread(target=entrenar_modelo, args=(update_rmsprop,))
#    thread.start()
#    return {"status": "entrenamiento_iniciado"}#

@fastapi_app.post("/entrenar")
async def iniciar_entrenamiento(req: Request):
    stop_flag["value"] = False
    body = await req.json()

    opt_name = body.get("opt", "rmsprop")
    num_epochs = body.get("num_epochs", 10)
    step_size = body.get("step_size", 0.001)
    alpha = body.get("alpha", 0.02)
    batch_size = body.get("batch_size", 32)
    r = body.get("r", 0.0)
    s = body.get("s", 0.0)
    beta = body.get("beta", 0.01)  # 🔹 nuevo parámetro para momentum

    optimizers = {
        "rmsprop": update_rmsprop,
        "sgd": update_sgd,
        "momentum": update_sgd_momentum,
        "adam": update_adam
    }

    selected_opt = optimizers.get(opt_name, update_rmsprop)

    if opt_name == "momentum":
        # 👉 Pasar beta como keyword argument específico
        thread = threading.Thread(
            target=entrenar_modelo,
            kwargs=dict(
                update=selected_opt,
                num_epochs=num_epochs,
                step_size=step_size,
                alpha_inicial=alpha,
                batch_size=batch_size,
                r=r,
                s=s,
                beta=beta  # ✅ solo en caso de momentum
            )
        )
    else:
        thread = threading.Thread(
            target=entrenar_modelo,
            args=(selected_opt, num_epochs, step_size, alpha, batch_size, r, s)
        )
    thread.start()
    return {"status": f"entrenamiento_iniciado_con_{opt_name}"}



#    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    # Lanza el servidor ASGI (Socket.IO + FastAPI)
def run_server():
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)


    # Si se ejecuta directamente como script
if __name__ == "__main__":
    # Hilo 1: servidor
    server_thread = threading.Thread(target=run_server)



    # Iniciar ambos hilos
    server_thread.start()


    # Esperar a que terminen (si querés)
    server_thread.join()
