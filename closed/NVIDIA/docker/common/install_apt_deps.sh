#!/bin/bash

set -e

if [[ $ARCH == "aarch64" ]]; then
    # Add mirror site as aarch64 pkgs sometimes get lost
	sed -i -e 's/http:\/\/archive/mirror:\/\/mirrors/' -e 's/\/ubuntu\//\/mirrors.txt/' /etc/apt/sources.list
else
    # MLPINF-1247 - Some partners in China are reporting DNS issues with Apt, specifically with cuda-repo. Remove the .list.
    rm -f /etc/apt/sources.list.d/cuda.list
fi


install_core_packages(){
    apt update
    apt install -y --no-install-recommends build-essential autoconf libtool git git-lfs \
        ccache curl wget pkg-config sudo ca-certificates automake libssl-dev tree \
        bc python3-dev python3-pip google-perftools gdb libglib2.0-dev clang sshfs libre2-dev \
        sysstat sshpass ntpdate less vim iputils-ping pybind11-dev
    apt install --only-upgrade libksba8
    apt remove -y cmake
    apt remove -y libgflags-dev
    apt remove -y libprotobuf-dev
    apt -y autoremove
    apt install -y --no-install-recommends pkg-config zip g++ zlib1g-dev unzip
    apt install -y --no-install-recommends libarchive-dev
    apt install -y --no-install-recommends rsync
}

install_platform_specific_x86_64(){
    apt install -y ripgrep
}

install_platform_specific_grace(){
    # Some convenient tools
    apt install -y ripgrep
}

install_platform_specific_soc(){
    apt update
    apt install --no-install-recommends -y moreutils rapidjson-dev libhdf5-dev \
        libgoogle-glog-dev libgflags-dev cmake libfreetype6-dev libpng-dev

    # Install nsys
    cd /tmp
    rm -f /usr/local/cuda/bin/nsys*
    wget --no-check-certificate https://nsys/dvs-prerel/QuadD_Auto/2025.4.2.56-254236150925v0/NsightSystems-cli-linux-drive-7-nda-arm64-DVS.deb
    dpkg -i NsightSystems-cli-linux-drive-7-nda-arm64-DVS.deb
    rm /tmp/*.deb
    cd -

    # Some convenient tools
    apt install -y ripgrep
}

case ${BUILD_CONTEXT} in
  x86_64)
    install_core_packages
    install_platform_specific_x86_64
    ;;
  aarch64-Grace)
    install_core_packages
    install_platform_specific_grace
    ;;
  aarch64-SoC)
    install_core_packages
    install_platform_specific_soc
    ;;
  *)
    echo "Supported BUILD_CONTEXT is only aarch64-SoC."
    exit 1
    ;;
esac
