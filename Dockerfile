FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY mlruns/ ./mlruns/

ENV MLFLOW_TRACKING_URI=file:/app/mlruns
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
