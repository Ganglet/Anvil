FROM python:3.11-slim

WORKDIR /app

# System deps for UMAP/HDBSCAN native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download model weights so the first audit request doesn't time out
RUN python -c "\
from torchvision.models import resnet18, ResNet18_Weights; \
resnet18(weights=ResNet18_Weights.IMAGENET1K_V1); \
from transformers import DistilBertForSequenceClassification; \
DistilBertForSequenceClassification.from_pretrained(\
'distilbert-base-uncased-finetuned-sst-2-english')"

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
