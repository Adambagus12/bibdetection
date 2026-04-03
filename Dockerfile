FROM python:3.10-slim

WORKDIR /app

ENV IS_SERVER=1

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    tesseract-ocr \
    tesseract-ocr-eng \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Buat folder upload
RUN mkdir -p /tmp/bib_uploads

# Konfigurasi supervisord
RUN echo "[supervisord]\nnodaemon=true\n\
[program:fastapi]\ncommand=uvicorn upload_server:app --host 0.0.0.0 --port 8000\n\
directory=/app\nautostart=true\nautorestart=true\n\
stdout_logfile=/dev/stdout\nstdout_logfile_maxbytes=0\n\
stderr_logfile=/dev/stderr\nstderr_logfile_maxbytes=0\n\n\
[program:streamlit]\ncommand=streamlit run scripts/app.py --server.port=8501 --server.address=0.0.0.0\n\
directory=/app\nautostart=true\nautorestart=true\n\
stdout_logfile=/dev/stdout\nstdout_logfile_maxbytes=0\n\
stderr_logfile=/dev/stderr\nstderr_logfile_maxbytes=0" > /etc/supervisor/conf.d/app.conf

EXPOSE 8501 8000

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]