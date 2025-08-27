FROM python:3.9-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/
COPY data/ /app/data/

# 确保数据目录存在
RUN mkdir -p /app/data

# 安装依赖
RUN pip install --no-cache-dir -r backend/requirements.txt

# 默认端口 (Hugging Face Spaces 使用 7860)
EXPOSE 7860

# 环境变量 (这个是占位符，实际值会从 Hugging Face Space 的 Repository secrets 中获取)
ENV DASHSCOPE_API_KEY=""

# 启动命令
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers"]
