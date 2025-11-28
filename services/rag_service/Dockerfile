
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY services/rag_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy all services so common_schemas is importable
COPY services /app/services

EXPOSE 8003

CMD ["uvicorn", "services.rag_service.app.main:app", "--host", "0.0.0.0", "--port", "8003"]
