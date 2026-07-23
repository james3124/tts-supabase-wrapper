FROM python:3.11-slim

WORKDIR /app

# Install deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Render injects PORT; default to 8000 for local dev
ENV PORT=8000

EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
