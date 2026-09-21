
FROM python:3.12-slim

# Install system dependencies (curl needed for healthcheck)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python deps first (layer caching optimisation) ─────
# If requirements.txt hasn't changed, Docker reuses this layer
# instead of reinstalling everything.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Download NLTK data at build time ───────────────────────────
RUN python -c "\
import nltk; \
nltk.download('punkt', quiet=True); \
nltk.download('punkt_tab', quiet=True); \
nltk.download('stopwords', quiet=True); \
nltk.download('wordnet', quiet=True); \
nltk.download('omw-1.4', quiet=True)"

# ── Copy application code ─────────────────────────────────────
COPY . .

# ── Create output directory ───────────────────────────────────
RUN mkdir -p output

# ── Expose Streamlit port ─────────────────────────────────────
EXPOSE 8501

# ── Entrypoint: run pipeline (if needed) then start dashboard ─
CMD sh -c "python run_pipeline.py && streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true"
