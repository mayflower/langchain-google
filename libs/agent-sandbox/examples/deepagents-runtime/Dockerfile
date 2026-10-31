FROM python:3.11-slim AS runtime

RUN useradd --create-home --uid 10001 appuser
WORKDIR /workspace

RUN pip install --no-cache-dir fastapi==0.125.0 uvicorn==0.38.0 python-multipart==0.0.21

COPY app.py /app/app.py
RUN chown -R appuser:appuser /workspace /app

USER appuser
ENV SANDBOX_RUNTIME_DIR=/workspace
EXPOSE 8080

CMD ["uvicorn", "app:app", "--app-dir", "/app", "--host", "0.0.0.0", "--port", "8080"]
