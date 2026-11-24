# syntax=docker/dockerfile:1.7
# ---- build stage: produce a wheel and install it into an isolated venv ----
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# Optional: behind a TLS-intercepting proxy, pass its CA bundle as a build secret:
#   docker build --secret id=ca,src=/path/to/ca-bundle.crt .
RUN --mount=type=secret,id=ca,required=false \
    if [ -s /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi \
 && python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip build \
 && /opt/venv/bin/python -m build --wheel --outdir /dist \
 && /opt/venv/bin/pip install "$(ls /dist/*.whl)[openai,gemini]" \
 && /opt/venv/bin/pip uninstall -y build

# ---- runtime stage: slim image, non-root user, no build tooling ----
FROM python:3.12-slim AS runtime
LABEL org.opencontainers.image.title="llmbench" \
      org.opencontainers.image.description="Evaluation, benchmarking and regression testing for LLM and RAG systems" \
      org.opencontainers.image.source="https://github.com/PiyushDas-2004/LLM-Bench" \
      org.opencontainers.image.licenses="MIT"
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LLMBENCH_DB_URL=sqlite:////data/llmbench.db \
    LLMBENCH_WORKSPACE=/app \
    LLMBENCH_LOG_FORMAT=json
RUN groupadd --system --gid 10001 llmbench \
 && useradd --system --uid 10001 --gid llmbench --home-dir /app --shell /usr/sbin/nologin llmbench \
 && mkdir -p /data /app \
 && chown llmbench:llmbench /data /app
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY --chown=llmbench:llmbench datasets ./datasets
COPY --chown=llmbench:llmbench examples ./examples
COPY --chown=llmbench:llmbench prompts ./prompts
COPY --chown=llmbench:llmbench rubrics ./rubrics
USER llmbench
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'), timeout=3)" || exit 1
ENTRYPOINT ["llmbench"]
# Default: serve the REST API on 0.0.0.0:$PORT (8000 if unset; requires LLMBENCH_API_KEY).
# Pass any other command to use the CLI, e.g. `docker run llmbench:local run examples/quickstart.yaml`.
CMD ["serve", "--host", "0.0.0.0"]
