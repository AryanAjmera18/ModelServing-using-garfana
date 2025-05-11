
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
from src.model import build_model
import io
import time
import mlflow.pytorch
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev only; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Prometheus metrics
PREDICTIONS_TOTAL = Counter("predictions_total", "Total prediction requests")
PREDICTION_ERRORS = Counter("prediction_errors_total", "Total failed prediction attempts")
INFERENCE_LATENCY = Histogram("inference_latency_seconds", "Time taken for a prediction")

# Load model
model = mlflow.pytorch.load_model("runs:/a9f25199ee7b4d78a7bae9df958e39e7/resnet50_custom_model")
model.eval()

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

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        start_time = time.time()
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        PREDICTIONS_TOTAL.inc()
        INFERENCE_LATENCY.observe(time.time() - start_time)

        response = {
            "predicted_class": class_map[pred],
            "confidence": round(confidence, 4)
        }
        return JSONResponse(content=response)

    except Exception as e:
        PREDICTION_ERRORS.inc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
