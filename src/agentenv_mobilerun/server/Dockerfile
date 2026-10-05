# MobileRun MCP server.
#
# Build context is the repo's `src/` directory, so the device client can be copied
# alongside the server:
#
#   agent-env env mcp-server put --id mobilerun \
#     --dockerfile src/agentenv_mobilerun/server/Dockerfile \
#     --context src
#
# `agent-env mobilerun setup` does exactly that for you.
#
# Pinned to the python:3.12-slim multi-arch index digest so a --platform build resolves
# the right architecture. To bump: docker manifest inspect python:3.12-slim
FROM python:3.12-slim@sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9

WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    MCP_PORT=18765 \
    MCP_HOST=0.0.0.0

COPY agentenv_mobilerun/server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Only the two modules the server actually imports. env.py / cli.py / steps/ are the
# agent-env side of the plugin and would drag agent-env into the image.
COPY agentenv_mobilerun/__init__.py agentenv_mobilerun/client.py /app/agentenv_mobilerun/
COPY agentenv_mobilerun/server/ /app/agentenv_mobilerun/server/

# Unprivileged: the server binds one high port, reads /app, and makes outbound HTTPS calls.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 18765
CMD ["python", "-m", "agentenv_mobilerun.server.main"]
