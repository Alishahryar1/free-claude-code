FROM ghcr.io/astral-sh/uv:0.5.18-python3.14-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8082

RUN groupadd -g 1000 app && useradd -u 1000 -g app user
USER user

CMD ["python", "server.py"]
