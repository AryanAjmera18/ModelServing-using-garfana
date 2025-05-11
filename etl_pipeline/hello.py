import requests

for i in range(20):
    with open("data/processed/Glaucoma/Glaucoma12.jpg", "rb") as f:
        response = requests.post("http://localhost:8000/predict", files={"file": f})
        print(i + 1, response.status_code, response.json())