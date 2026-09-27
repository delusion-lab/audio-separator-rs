# audio-separator-server 多阶段构建（Linux / amd64）
# 包含 ONNX Runtime Linux 库，同时支持 local 与 mvsep 后端。

FROM rust:1.85-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 官方 ONNX Runtime Linux 动态库（与 scripts/fetch-ort.ps1 同版本 1.28）
RUN curl -fsSL -o /tmp/ort.tgz \
      https://github.com/microsoft/onnxruntime/releases/download/v1.28.0/onnxruntime-linux-x64-1.28.0.tgz \
    && mkdir -p /opt/ort \
    && tar -xzf /tmp/ort.tgz -C /opt/ort --strip-components=1 \
    && rm /tmp/ort.tgz \
    && ls /opt/ort/lib

WORKDIR /app

# 先复制清单与依赖声明以利用缓存
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY models.json ./models.json

# 生成本机 ORT 配置（等价于 Windows 侧 .cargo/config.toml）
RUN mkdir -p .cargo \
    && printf '[env]\nORT_LIB_PATH = { value = "/opt/ort/lib", force = true }\n' > .cargo/config.toml

RUN cargo build --release -p audio-separator-server

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libasound2 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ONNX Runtime 动态库
COPY --from=builder /opt/ort/lib/libonnxruntime.so* /usr/lib/

# 服务端二进制
COPY --from=builder /app/target/release/audio-separator-server /usr/local/bin/audio-separator-server

# MVSEP API Key（运行时通过环境变量注入）
ENV ASEP_MVSEP_API_KEY=

EXPOSE 8080

ENTRYPOINT ["audio-separator-server"]
CMD ["--addr", "0.0.0.0:8080"]
