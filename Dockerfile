FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim

WORKDIR /app
COPY ./ ./

RUN uv sync --no-dev

ENV PYTHONUNBUFFERED=1

EXPOSE 8082

CMD ["uv", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8082"]
