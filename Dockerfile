# NOTE: This is the runtime cuDNN image. It does not contain the development tools.
#       In case you want to run a different version of PyTorch or CUDA, 
#       see https://hub.docker.com/r/pytorch/pytorch/tags
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
