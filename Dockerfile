FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential cmake curl unzip pkg-config \
    libopencv-dev \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libspdlog-dev \
    && rm -rf /var/lib/apt/lists/*

# Download libtorch C++ distribution (CPU)
RUN curl -L \
 https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcpu.zip \
    -o /tmp/libtorch.zip && \
    unzip /tmp/libtorch.zip -d /opt && \
    rm /tmp/libtorch.zip

WORKDIR /app

# Copy sources
COPY CMakeLists.txt .
COPY src/ ./src
COPY dataset/ ./dataset
COPY models/ ./models

# Tell CMake where torch is
ENV CMAKE_PREFIX_PATH=/opt/libtorch
