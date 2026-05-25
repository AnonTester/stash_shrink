ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

EXPOSE 9899

CMD ["python", "stash_shrink.py"]
