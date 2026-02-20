FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project or just necessary parts
COPY src/ src/
COPY api/ api/
COPY models/ models/
# In a real scenario, we might pull models from DVC/cloud storage here
# For this assignment, we assume models are present (or we mount them)

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]