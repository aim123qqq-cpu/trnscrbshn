FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=10000 \
    DEFAULT_BACKEND=openai \
    WORK_DIR=/tmp/transcriber/jobs \
    OUTPUT_DIR=/tmp/transcriber/outputs

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py README.md ./

EXPOSE 10000

CMD ["python", "app.py"]
