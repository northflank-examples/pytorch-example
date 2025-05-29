# PyTorch Fashion MNIST FastAPI

A FastAPI web service that implements a PyTorch neural network for Fashion MNIST classification.

## Installation

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check

- **GET** `/` - Check if the service is running

### Model Information

- **GET** `/model/info` - Get information about the current model

### Training

- **POST** `/train` - Train the model

**Request Body:**

```json
{
  "epochs": 5,
  "learning_rate": 0.001,
  "batch_size": 64
}
```

**Response:**

```json
{
  "message": "Training completed successfully",
  "epochs_completed": 5,
  "final_accuracy": 0.85,
  "final_loss": 0.45
}
```

### Prediction

- **POST** `/predict` - Make a prediction on an image

**Request Body:**

```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAABU0lEQVR4nN2RPUvDUBiFz3tzm3iTNK2EalXwYyhFRZxEsbiJ7uLi4GIXf4D/S3Bw8hc4OhVcFBWtaGqbpprkfkgRhHZ38Uwv5+G8HDjAv5ZFgIMawEZsYpg79JQBUhyMQ2jsNJrDY+o4BvjoQ7mx3K6dR+I+DB7Hkkx6hzRRJEareYf/QgIjWBqn7Rcx0VY6yezA+4EEA20shaPqS7/8/przAphX+4EGjBMUTtYfwrLoFfUgJ4P9YSEGQ1oDswfi1nfCzLhQqUp0A9xSehitLNZnsl45KDh6ofCRa/ZpxatcYXrB88SSm/dZSUh3kNrPJbfjTybVkGN3VlYsJWO/Sk6H+VbS704BHS0cyfeareeYZRbFti6S0IXq9IrNkLhf/Vd+vbXWgIyjqGtTWHcDs35zt+sYyKdekQD4m/XtSuCR0VHr6vILuJh/i2OZnv3J5n+mb2KCff/l9FsMAAAAAElFTkSuQmCC"
}
```

**Response:**

```json
{
  "predicted_class": "Ankle boot",
  "confidence": 0.3751780688762665,
  "probabilities": [
    0.00274666422046721, 0.0014659167500212789, 0.007203937042504549,
    0.0027840372640639544, 0.006614977028220892, 0.21854005753993988,
    0.007174666039645672, 0.2593027353286743, 0.1189890205860138,
    0.3751780688762665
  ]
}
```

## Fashion MNIST Classes

The model can classify images into these 10 categories:

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

## Interactive Documentation

Once the server is running, you can access:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Example Usage

### Training the Model

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{"epochs": 3, "learning_rate": 0.001, "batch_size": 64}'
```

### Making Predictions

To make a prediction, you need to encode your image (28x28 grayscale) as base64:

```python
import base64
from PIL import Image
import io

# Load and encode image
with open("fashion_item.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

# Make request
import requests
response = requests.post(
    "http://localhost:8000/predict",
    json={"image_base64": encoded_string}
)
print(response.json())
```

## Notes

- Images for prediction should be 28x28 grayscale images (like Fashion MNIST)
- The API will automatically resize and convert images if needed
- The model is automatically saved to `model.pth` after training
- If `model.pth` exists on startup, it will be loaded automatically