FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        fastapi \
        uvicorn \
        pydantic \
        pydantic-settings \
        numpy \
        ruamel.yaml \
        pyarabic \
        joblib

# Copy code and assets
COPY services/common_schemas ./services/common_schemas
COPY Models/Ashaar_runtime/ashaar_only ./models/Ashaar_runtime/ashaar_only
COPY services/ashaar_meter_service ./services/ashaar_meter_service

ENV PYTHONPATH=/app

EXPOSE 8005

CMD ["uvicorn", "services.ashaar_meter_service.app.main:app", "--host", "0.0.0.0", "--port", "8005"]
