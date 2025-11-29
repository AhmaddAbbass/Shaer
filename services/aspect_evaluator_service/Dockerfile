FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal runtime deps
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        fastapi \
        uvicorn \
        httpx \
        pydantic \
        pydantic-settings \
        openai

# Copy shared schemas and the service code
COPY services/common_schemas ./services/common_schemas
COPY services/aspect_evaluator_service ./services/aspect_evaluator_service

ENV PYTHONPATH=/app

EXPOSE 8106

CMD ["uvicorn", "services.aspect_evaluator_service.app.main:app", "--host", "0.0.0.0", "--port", "8106"]
