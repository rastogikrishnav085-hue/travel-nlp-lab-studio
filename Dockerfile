# Travel Query Intent Classification Using NLP — Travel NLP Lab Studio (Streamlit)
# Build:  docker build -t travel-nlp-studio .
# Run:    docker run --rm -p 8501:8501 travel-nlp-studio      then open http://localhost:8501
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install the Studio's runtime packages first (cached between builds when only code changes)
COPY requirements.txt .
RUN pip install -r requirements.txt

# App code, example run bundle, raw datasets, notebook, fonts and plotly.js (everything works offline)
COPY . .

# Run as a normal user, not root
RUN useradd --create-home studio && chown -R studio:studio /app
USER studio

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')" || exit 1

# Inside a container the app must listen on all interfaces so the published port reaches it
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
