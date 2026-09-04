FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HGAC_PUBLIC_MODE=1 \
    HGAC_RUNTIME_ROOT=/var/data

RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r /app/requirements.txt

COPY . /app

RUN mkdir -p /var/data/data/regional_cache/catalog \
    /var/data/data/regional_cache/dem \
    /var/data/workspaces

EXPOSE 10000

CMD ["python", "app.py"]
