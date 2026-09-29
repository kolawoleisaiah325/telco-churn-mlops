FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHURN_MODEL_PATH=/app/model/churn_model.joblib

WORKDIR /app

LABEL org.opencontainers.image.source="https://github.com/kolawoleisaiah325/telco-churn-mlops" \
      org.opencontainers.image.description="Telco churn prediction API"

# XGBoost uses the OpenMP runtime on Linux.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serving.txt ./
RUN pip install --no-cache-dir -r requirements-serving.txt

RUN useradd --no-log-init --create-home --uid 10001 appuser
COPY api.py ./
COPY churn/ ./churn/
COPY model/churn_model.joblib ./model/churn_model.joblib

USER appuser
EXPOSE 8080

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
