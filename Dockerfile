FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx /bin/
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev
COPY data ./data

EXPOSE 8501 8000
CMD ["uv", "run", "streamlit", "run", "src/bizcompass_kh/ui/app.py", "--server.address=0.0.0.0"]
