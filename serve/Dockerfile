FROM python:3.10-slim
WORKDIR /app
COPY serve/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY . /app
ENV MODEL_PATH=/app/runs/model_fused.pth
EXPOSE 8000
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
