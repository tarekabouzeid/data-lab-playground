# Airflow MCP server (Astronomer astro-airflow-mcp) for the Helm chart (airflowMcp.enabled).
# Baked into an image so pods start without reaching PyPI.
FROM python:3.12.15-slim-bookworm

ARG ASTRO_AIRFLOW_MCP_VERSION=0.9.1

RUN pip install --no-cache-dir "astro-airflow-mcp==${ASTRO_AIRFLOW_MCP_VERSION}" \
    && useradd --uid 1000 --create-home --shell /usr/sbin/nologin mcp

# HTTP transport on 0.0.0.0:8000 (/mcp). Telemetry to Astronomer is off.
ENV MCP_TRANSPORT=http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000 \
    AF_TELEMETRY_DISABLED=true \
    AF_READ_ONLY=true

USER 1000
EXPOSE 8000
CMD ["astro-airflow-mcp"]
