ARG BASE_IMAGE
FROM ${BASE_IMAGE} as base

ARG ENV
ARG BUILD_CONTEXT
ARG MITTEN_HASH

ENV ENV=${ENV}
ENV BUILD_CONTEXT=${BUILD_CONTEXT}
ENV MITTEN_GIT_URL=ssh://git@gitlab-master.nvidia.com:12051/adlsa/MLPerf-Automotive/mlpauto-mitten.git
ENV MITTEN_HASH=${MITTEN_HASH}

SHELL ["/bin/bash", "-c"]

WORKDIR /tmp

COPY --from=common /requirements.${BUILD_CONTEXT}.* .

# 1. apt installs
COPY --from=common /install_apt_deps.sh .
RUN if [ -f install_apt_deps.sh ]; then bash install_apt_deps.sh; fi

# 2. pip installs
COPY --from=common /install_pip_deps.sh .
RUN if [ -f install_pip_deps.sh ]; then bash install_pip_deps.sh; fi

# 3. Any pip install or from-source build required.
COPY --from=common /install_misc_deps.sh .
RUN if [ -f install_misc_deps.sh ]; then bash install_misc_deps.sh; fi

# 4. install docker-within-docker
COPY --from=common /install_docker.sh .
RUN if [ -f install_docker.sh ]; then bash install_docker.sh; fi

# 5. install mlperf mitten
RUN git clone ${MITTEN_GIT_URL} /tmp/mitten
COPY --from=common /install_mitten.sh .
RUN if [ -f install_mitten.sh ]; then bash install_mitten.sh; fi

WORKDIR /work
