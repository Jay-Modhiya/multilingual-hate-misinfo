FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ git curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY configs/ configs/
COPY src/ src/
COPY app/ app/
RUN mkdir -p models/saved outputs data/raw mlruns logs
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 CMD curl -f http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
