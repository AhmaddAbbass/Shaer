FROM python:3.11-slim

WORKDIR /app

COPY services/yehia_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt

# Copy shared schemas and service code
COPY services/common_schemas ./services/common_schemas
COPY services/yehia_service/app /app/app

ENV PYTHONPATH=/app
EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
