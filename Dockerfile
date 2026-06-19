FROM python:3.11-slim

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install dependencies first (better layer cache when only code changes).
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --group test

COPY tests ./tests
COPY configs ./configs

CMD ["uv", "run", "pytest"]
