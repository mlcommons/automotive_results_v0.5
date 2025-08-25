# syntax=docker/dockerfile:1.4

ARG BASE_IMAGE
FROM ${BASE_IMAGE} as base

ARG ENV
ARG BUILD_CONTEXT
ARG CUDA_VER
ARG TRT_MAJOR_VER
ARG TRT_VER
ARG USE_NIGHTLY
ARG MITTEN_HASH

ENV ENV=${ENV}
ENV BUILD_CONTEXT=${BUILD_CONTEXT}
ENV CUDA_VER=${CUDA_VER}
ENV TRT_MAJOR_VER=${TRT_MAJOR_VER}
ENV TRT_VER=${TRT_VER}
ENV USE_NIGHTLY=${USE_NIGHTLY}
ENV USE_DEB_PKG=${USE_DEB_PKG}

SHELL ["/bin/bash", "-c"]

WORKDIR /tmp

# 1. apt installs
COPY --from=common /install_apt_deps.sh .
RUN if [ -f install_apt_deps.sh ]; then bash install_apt_deps.sh; fi

# 2. pip installs
COPY --from=common /requirements.${BUILD_CONTEXT}.* .
COPY --from=common /install_pip_deps.sh .
RUN if [ -f install_pip_deps.sh ]; then bash install_pip_deps.sh; fi

# 3. Any from-source build required.
COPY --from=common /install_misc_deps.sh .
RUN if [ -f install_misc_deps.sh ]; then bash install_misc_deps.sh; fi

# 4. install docker-within-docker
COPY --from=common /install_docker.sh .
RUN if [ -f install_docker.sh ]; then bash install_docker.sh; fi

# 5. install TensorRT for dev mode
COPY install_tensorrt.sh .
RUN if [ -f install_tensorrt.sh ]; then bash install_tensorrt.sh; fi

# 6. install mlperf mitten
COPY --from=common /mlpauto-mitten /tmp/mitten
COPY --from=common /install_mitten.sh .
RUN if [ -f install_mitten.sh ]; then bash install_mitten.sh; fi

# Need to export LD_LIBRARY_PATH for thor packages
# FIXME: any way to make this more portable?
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.8/thor/targets/aarch64-linux/lib

WORKDIR /work
