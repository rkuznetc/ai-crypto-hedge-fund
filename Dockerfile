FROM python:3.11-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
COPY src ./src
COPY tests ./tests
COPY configs ./configs
COPY data ./data

RUN uv sync --frozen --group dev

CMD ["uv", "run", "pytest"]
