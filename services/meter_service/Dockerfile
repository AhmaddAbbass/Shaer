FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python deps (minimal set needed for meter service)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        fastapi \
        uvicorn \
        pydantic \
        pydantic-settings \
        "tensorflow-cpu==2.12.0" \
        scikit-learn \
        numpy \
        joblib \
        h5py

# Copy code and assets
COPY services/common_schemas ./services/common_schemas
COPY utils ./utils
COPY models/Ashaar_runtime/bilstm_model ./models/Ashaar_runtime/bilstm_model
COPY services/meter_service ./services/meter_service

ENV PYTHONPATH=/app

EXPOSE 8004

CMD ["uvicorn", "services.meter_service.app.main:app", "--host", "0.0.0.0", "--port", "8004"]
