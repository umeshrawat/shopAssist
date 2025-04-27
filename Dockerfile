# Use lightweight Python image
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src
COPY data/ ./data

EXPOSE 8000

CMD ["uvicorn", "src.inference_api.inference:app", "--host", "0.0.0.0", "--port", "8000"]
