# Loosely based on https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from typing import List
import base64
import io
from PIL import Image
import numpy as np

app = FastAPI(title="PyTorch Fashion MNIST API", version="1.0.0")

model = None
device = None
classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

class TrainRequest(BaseModel):
    epochs: int = 5
    learning_rate: float = 1e-3
    batch_size: int = 64

class PredictRequest(BaseModel):
    image_base64: str

class TrainResponse(BaseModel):
    message: str
    epochs_completed: int
    final_accuracy: float
    final_loss: float

class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: List[float]

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def initialize_model():
    global model, device
    
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    
    model = NeuralNetwork().to(device)
    
    if os.path.exists("model.pth"):
        model.load_state_dict(torch.load("model.pth", weights_only=True, map_location=device))
        print("Loaded existing model from model.pth")

def get_data_loaders(batch_size: int = 64):
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    
    return train_dataloader, test_dataloader

def train_epoch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        pred = model(X)
        loss = loss_fn(pred, y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
        if batch % 100 == 0:
            loss_value, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")
    
    return total_loss / len(dataloader)

def test_model(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    accuracy = correct / size
    
    return accuracy, test_loss

def preprocess_image(image_base64: str):
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'L':
            image = image.convert('L')
        
        if image.size != (28, 28):
            image = image.resize((28, 28))
        
        image_array = np.array(image) / 255.0
        tensor = torch.from_numpy(image_array).float().unsqueeze(0).unsqueeze(0)
        
        return tensor.to(device)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.on_event("startup")
async def startup_event():
    initialize_model()

@app.get("/")
async def root():
    return {"message": "PyTorch Fashion MNIST API is running", "model_loaded": model is not None}

@app.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    global model
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        print(f"Starting training with {request.epochs} epochs...")
        
        train_dataloader, test_dataloader = get_data_loaders(request.batch_size)
        
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=request.learning_rate)
        
        final_accuracy = 0.0
        final_loss = 0.0
        
        for epoch in range(request.epochs):
            print(f"Epoch {epoch + 1}/{request.epochs}")
            print("-" * 30)
            
            train_epoch(train_dataloader, model, loss_fn, optimizer)
            
            accuracy, test_loss = test_model(test_dataloader, model, loss_fn)
            
            print(f"Test Error: Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f}\n")
            
            final_accuracy = accuracy
            final_loss = test_loss
        
        torch.save(model.state_dict(), "model.pth")
        print("Model saved to model.pth")
        
        return TrainResponse(
            message="Training completed successfully",
            epochs_completed=request.epochs,
            final_accuracy=final_accuracy,
            final_loss=final_loss
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    global model
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        image_tensor = preprocess_image(request.image_base64)
        
        model.eval()
        with torch.no_grad():
            logits = model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class_idx = logits.argmax(1).item()
            confidence = probabilities[0][predicted_class_idx].item()
        
        return PredictResponse(
            predicted_class=classes[predicted_class_idx],
            confidence=confidence,
            probabilities=probabilities[0].tolist()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    global model, device
    
    if model is None:
        return {"status": "Model not initialized"}
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "status": "Model initialized",
        "device": device,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_file_exists": os.path.exists("model.pth"),
        "classes": classes
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
