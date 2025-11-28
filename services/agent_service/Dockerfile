FROM python:3.11-slim

WORKDIR /app

COPY services/agent_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY services/agent_service /app/services/agent_service
COPY services/common_schemas /app/services/common_schemas

CMD ["uvicorn", "services.agent_service.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
