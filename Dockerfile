FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

COPY . .

RUN mkdir -p output \
    && useradd --create-home --shell /usr/sbin/nologin --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

CMD ["sh", "-c", "python run_pipeline.py && streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true"]
