FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG WORKSPACE=/workspace
ARG MINICONDA_VERSION=py312_25.1.1-2
ARG DOTNET_SDK_VERSION=9.0
ARG CODE_SERVER_VERSION=4.101.2

ENV CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:/usr/local/bin:/usr/share/dotnet:${PATH} \
    DOTNET_ROOT=/usr/lib/dotnet \
    DOTNET_NOLOGO=true \
    DOTNET_CLI_TELEMETRY_OPTOUT=true

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Core OS, build, graphics, Wayland, X11, OpenGL, and PicoGK/OpenVDB deps.
RUN apt-get update \
    && apt-get install -y software-properties-common  \
    && apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        wget \
        git \
        git-lfs \
        sudo \
        locales \
        less \
        nano \
        pkg-config \
        build-essential \
        gcc \
        g++ \
        gdb \
        cmake \
        ninja-build \
        make \
        tar \
        bzip2 \
        unzip \
        xz-utils \
        python3 \
        python3-pip \
        python3-venv \
        libboost-all-dev \
        libtbb-dev \
        libblosc-dev \
        zlib1g-dev \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        mesa-common-dev \
        libglfw3-dev \
        libx11-dev \
        libxrandr-dev \
        libxinerama-dev \
        libxcursor-dev \
        libxi-dev \
        libxkbcommon-dev \
        libwayland-dev \
        wayland-protocols \
        xorg-dev \
        libgtk-3-0 \
        libnss3 \
        libxss1 \
        libasound2t64 \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Browser-accessible VS Code-compatible remote application.
RUN curl -fsSL "https://github.com/coder/code-server/releases/download/v${CODE_SERVER_VERSION}/code-server_${CODE_SERVER_VERSION}_amd64.deb" -o /tmp/code-server.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends /tmp/code-server.deb \
    && rm -f /tmp/code-server.deb \
    && rm -rf /var/lib/apt/lists/*

# Conda
RUN wget -q "https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh" -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniconda.sh \
    && conda config --system --set auto_update_conda false \
    && conda config --system --add channels defaults \
    && conda config --system --add channels conda-forge \
    && conda config --system --set channel_priority strict \
    && conda update -y -n base conda

COPY environment.OFTPMSoptimiser.yml /tmp/OFTPMSoptimiser.yml
RUN conda env create -f /tmp/OFTPMSoptimiser.yml python=3.12 \
    && rm /tmp/OFTPMSoptimiser.yml \
    && conda clean -afy

RUN useradd -m -s /bin/bash vscode \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/vscode \
    && chmod 0440 /etc/sudoers.d/vscode \
    && install -d -o vscode -g vscode "${WORKSPACE}"

USER vscode

WORKDIR ${WORKSPACE}

RUN git clone https://github.com/HuiLucas/MTO.git

RUN conda init bash \
    && echo "conda activate OFTPMSoptimiser" >> /home/vscode/.bashrc

USER root

RUN chmod -R g+w /opt/conda

RUN sh -c "wget -O - https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc" \
    && add-apt-repository http://dl.openfoam.org/ubuntu

RUN apt update && apt -y install openfoam11

SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/openfoam11/etc/bashrc" >> /etc/bash.bashrc

USER vscode

EXPOSE 8080, 50051

CMD ["/bin/bash"]
