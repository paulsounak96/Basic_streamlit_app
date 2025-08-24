from fastapi import FastAPI
from pydantic import BaseModel
import torch
from model import FeedForwardNet

class InputData(BaseModel):
    features: list[float]

app = FastAPI()

# Model params must match train.py
input_dim, hidden_dim, output_dim = 4, 64, 2
model = FeedForwardNet(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

@app.post("/predict")
def predict(data: InputData):
    x = torch.tensor(data.features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).tolist()[0]
        pred_class = int(torch.argmax(logits))
    return {"probabilities": probs, "predicted_class": pred_class}
