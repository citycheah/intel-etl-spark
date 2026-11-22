FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    openjdk-17-jre-headless \
    procps \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /workspace/intel-etl-spark
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY intel_etl_spark ./intel_etl_spark
COPY scripts ./scripts
COPY tests ./tests

RUN uv sync --extra dev --no-cache

CMD ["uv", "run", "pytest"]
