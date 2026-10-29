FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 单镜像多服务：镜像里不放业务角色，由 command 决定起哪个 Agent（对应 P11 模式）
RUN pip install --no-cache-dir \
    "a2a-sdk[http-server]==0.3.22" \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.30" \
    "httpx>=0.28"

COPY config.py ./
COPY app ./app

EXPOSE 8201

CMD ["python", "-m", "app.runtime", "cards"]
