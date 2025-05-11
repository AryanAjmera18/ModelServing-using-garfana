from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image
import io
import numpy as np
import onnxruntime as ort
import time
import os
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from torchvision import transforms

app = FastAPI()

# Prometheus metrics
PREDICTIONS_TOTAL = Counter("predictions_total", "Total prediction requests")
PREDICTION_ERRORS = Counter("prediction_errors_total", "Total failed prediction attempts")
INFERENCE_LATENCY = Histogram("inference_latency_seconds", "Time taken for a prediction")
MODEL_CONFIDENCE = Gauge("model_confidence_score", "Confidence of last prediction")
DRIFT_ALERT = Gauge("drift_alert", "1 if low confidence drift detected")

low_confidence_streak = 0

# Load ONNX model with dynamic path
model_path = os.path.join(os.getcwd(), "resnet50_custom_model.onnx")
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# Class map
class_map = {
    0: "Central Serous Chorioretinopathy",
    1: "Diabetic Retinopathy",
    2: "Disc Edema",
    3: "Glaucoma",
    4: "Healthy",
    5: "Macular Scar",
    6: "Myopia",
    7: "Pterygium",
    8: "Retinal Detachment",
    9: "Retinitis Pigmentosa"
}

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global low_confidence_streak
    try:
        start = time.time()
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).numpy()

        outputs = session.run(None, {input_name: input_tensor})
        probs = np.exp(outputs[0]) / np.sum(np.exp(outputs[0]), axis=1, keepdims=True)
        pred = int(np.argmax(probs))
        confidence = float(probs[0][pred])

        PREDICTIONS_TOTAL.inc()
        INFERENCE_LATENCY.observe(time.time() - start)
        MODEL_CONFIDENCE.set(confidence)

        if confidence < 0.5:
            low_confidence_streak += 1
            if low_confidence_streak >= 5:
                DRIFT_ALERT.set(1)
        else:
            low_confidence_streak = 0
            DRIFT_ALERT.set(0)

        return JSONResponse({
            "predicted_class": class_map[pred],
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        PREDICTION_ERRORS.inc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
