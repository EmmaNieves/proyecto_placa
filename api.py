from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from processor import detectar_placa
from database import save_plate, get_all_plates

app = FastAPI()


@app.get("/")
def root():
    return {"status": "ok", "mensaje": "Sistema de placas activo"}


@app.post("/detectar")
async def detectar(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Imagen inválida"}

    placas = detectar_placa(frame)

    if not placas:
        return {"resultado": "sin_deteccion", "placas": [], "registrada": False}

    placas_detectadas = []
    for p in placas:
        is_new = save_plate(p["placa"], frame)
        placas_detectadas.append({
            **p,
            "es_nueva": is_new,
            "registrada": not is_new
        })
        print(f"{'✅ Nueva' if is_new else '🔁 Registrada'}: {p['placa']}")

    return {
        "resultado": "ok",
        "placas": placas_detectadas,
        "registrada": any(p["registrada"] for p in placas_detectadas)
    }


@app.get("/placas")
def listar_placas():
    return {"placas": get_all_plates()}