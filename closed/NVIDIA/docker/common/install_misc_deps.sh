#!/bin/bash

set -e

install_gflags(){
    local VERSION=$1

    cd /tmp
    # -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON -DBUILD_gflags_LIB=ON .. \
    git clone -b v${VERSION} https://github.com/gflags/gflags.git
    cd gflags
    mkdir build && cd build
    cmake -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON -DBUILD_gflags_LIB=ON ..
    make -j
    make install
    cd /tmp && rm -rf gflags
}

install_glog(){
    local VERSION=$1

    cd /tmp
    git clone -b v${VERSION} https://github.com/google/glog.git
    cd glog
    cmake -H. -Bbuild -G "Unix Makefiles" -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON
    cmake --build build
    cmake --build build --target install
    cd /tmp && rm -rf glog

}

install_nlohmann_json_x86_64(){
    # Install nlohmann/json for LLM config parsing
    cd /tmp
    git clone -b v3.11.2 https://github.com/nlohmann/json.git
    cp -r json/single_include/nlohmann /usr/include/x86_64-linux-gnu/
    rm -rf json
}

install_nlohmann_json_aarch64(){
    apt install -y nlohmann-json3-dev
}

install_nvrtc_dev(){
    local ARCH=$1
    case "$ARCH" in
        x86_64)
            PKG_ARCH="amd64"
            ;;
        sbsa|aarch64)
            PKG_ARCH="arm64"
            ;;
        *)
            echo "Unsupported ARCH: $ARCH"
            exit 1
            ;;
    esac

    cd /tmp
    FULL_CUDA_VERSION=$(nvcc --version | grep release | awk '{print $6}' | tr -d 'V')
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${ARCH}/cuda-nvrtc-dev-$(echo $CUDA_VER | sed 's/\./-/g')_${FULL_CUDA_VERSION}-1_${PKG_ARCH}.deb -O /tmp/nvrtc.deb
    dpkg -i /tmp/nvrtc.deb
}

case ${BUILD_CONTEXT} in
  x86_64)
    install_gflags 2.2.1
    install_glog 0.6.0
    install_nlohmann_json_x86_64
    install_nvrtc_dev x86_64
    ;;
  aarch64-Grace)
    install_gflags 2.2.2
    install_glog 0.6.0
    install_nlohmann_json_aarch64
    install_nvrtc_dev sbsa
    ;;
  aarch64-SoC)
    install_gflags 2.2.2
    install_glog 0.6.0
    install_nlohmann_json_aarch64
    ;;
  *)
    echo "Supported BUILD_CONTEXT is only aarch64-SoC."
    exit 1
    ;;
esac
