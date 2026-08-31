FROM python:3.11-slim

# System deps for opencv/pillow used by ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/

# Run as a non-root user in the final image rather than default root.
RUN useradd --create-home --uid 1000 wastevision \
    && chown -R wastevision:wastevision /app
USER wastevision

ENV WASTEVISION_WEIGHTS=/app/models/best.pt
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
