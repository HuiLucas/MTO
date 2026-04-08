FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    gnupg2 \
    lsb-release \
    wget \
    software-properties-common \
    build-essential \
    cmake \
    make \
    gfortran \
    openmpi-bin \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q -O - https://dl.openfoam.org/gpg.key \
    | gpg --dearmor -o /usr/share/keyrings/openfoam.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/openfoam.gpg] http://dl.openfoam.org/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME}) main" \
    > /etc/apt/sources.list.d/openfoam.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends openfoam11 \
    && rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/openfoam11/etc/bashrc" >> /etc/bash.bashrc
