# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=11.7.1


FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

ARG PYTHON_VERSION=3.10
ARG PYTORCH_VERSION=2.0

SHELL ["/bin/bash", "-c"]
# https://techoverflow.net/2019/05/18/how-to-fix-configuring-tzdata-interactive-input-when-building-docker-images/
ENV \
    DEBIAN_FRONTEND="noninteractive" \
    TZ="Etc/UTC" \
    PATH="$PATH:/root/.local/bin" \
    CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda" \
    MKL_THREADING_LAYER="GNU" \
    # MAKEFLAGS="-j$(nproc)"
    MAKEFLAGS="-j2"

RUN \
    apt-get -y update --fix-missing && \
    apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages \
        build-essential \
        pkg-config \
        cmake \
        git \
        wget \
        curl \
        zip \
        unzip \
        g++ \
        cmake \
        ffmpeg \
        git \
        libsndfile1 \
        ca-certificates \
        software-properties-common \
        libopenmpi-dev \
        openmpi-bin \
        ssh \
    && \
    # Install python
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-distutils \
        python${PYTHON_VERSION}-dev \
    && \
    update-alternatives --install /usr/bin/python${PYTHON_VERSION%%.*} python${PYTHON_VERSION%%.*} /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    curl https://bootstrap.pypa.io/get-pip.py | python && \
    # Cleaning
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH="/usr/lib/python${PYTHON_VERSION}/site-packages"

COPY requirements/ requirements/
COPY .github/assistant.py ./assistant.py

RUN \
    # set particular PyTorch version
    pip install -q wget packaging "setuptools==69.5.1" && \
    python -m wget https://raw.githubusercontent.com/Lightning-AI/utilities/main/scripts/adjust-torch-versions.py  && \
    for fpath in `ls requirements/*.txt`; do \
      python ./adjust-torch-versions.py $fpath ${PYTORCH_VERSION}; \
    done && \
    # trying to resolve pesq installation issue
    pip install -q "numpy<1.24" && \
    CUDA_VERSION_MM=${CUDA_VERSION%.*} && \
    CU_VERSION_MM=${CUDA_VERSION_MM//'.'/''} && \
    # requirements for assistant
    pip install -q -U packaging fire && \
    # switch some packages to be GPU related
    python assistant.py replace_str_requirements "onnxruntime" "onnxruntime_gpu" --req_files requirements/audio.txt && \
    # install develop environment
    pip install --no-cache-dir -r requirements/_devel.txt \
      --extra-index-url="https://download.pytorch.org/whl/cu${CU_VERSION_MM}/" \
      --extra-index-url="https://download.pytorch.org/whl/test/cu${CU_VERSION_MM}" && \
    rm -rf requirements/

RUN \
    # Show what we have
    pip --version && \
    pip list && \
    python -c "import sys; ver = sys.version_info ; assert f'{ver.major}.{ver.minor}' == '$PYTHON_VERSION', ver" && \
    python -c "import torch; assert torch.__version__.startswith('$PYTORCH_VERSION'), torch.__version__"
