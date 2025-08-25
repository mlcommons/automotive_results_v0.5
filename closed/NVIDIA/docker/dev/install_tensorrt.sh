#!/bin/bash

set -e

install_deb_pkg() {
    wget $TRT_DEB_URL/$1 -O $1 && dpkg -i $1 && rm $1;
}


install_tensorrt_x86_64_nightly(){
  apt remove -y tensorrt-dev libnvparsers-dev
  rm -rf /usr/local/lib/python3.10/dist-packages/tensorrt*
  export TRT_NIGHTLY_URL=http://cuda-repo-k8s/release-candidates/Libraries/TensorRT/v10.13/10.13.0.27-f62a134b/12.8-r570/d7l-aarch64/tar/TensorRT-10.13.0.27.d7l.aarch64-gnu.cuda-12.8.tar.gz
  cd /tmp
  wget ${TRT_NIGHTLY_URL} --user tensorrt-read-only --password "Tensorrt@123" -O TRT.tar
  tar -xf TRT.tar
  cd source/install/x86_64-gnu
  mkdir trt
  tar -xvzf cuda-${CUDA_VER}/release_tarfile/TensorRT-*.*.x86_64-gnu.cuda-${CUDA_VER}.tar.gz -C trt --strip-components 1
  tar -xvzf TensorRT-*.*.x86_64-gnu.cuda-${CUDA_VER}.internal.tar.gz -C trt --strip-components 1
  cp -rv trt/lib/* /usr/lib/x86_64-linux-gnu/
  cp -rv trt/include/* /usr/include/x86_64-linux-gnu/
  cp -rv trt/bin/* /usr/bin/
  python3 -m pip install trt/python/tensorrt-*-cp310-none-linux_x86_64.whl
  cd ../../..
  rm -rf source
  rm -f TRT.tar
}

install_tensorrt_x86_64(){
  if [[ $(apt list --installed | grep libnvinfer) ]]; then apt-get remove --purge -y libnvinfer*; fi
  if [[ $(apt list --installed | grep tensorrt) ]]; then apt-get remove --purge -y tensorrt*; fi
  pip3 uninstall -y tensorrt

  PY_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[0:2])))')
  PARSED_PY_VERSION=$(echo "${PY_VERSION//./}")

  # install tensorrt
  RELEASE_URL_TRT="https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.8.0/tars/TensorRT-10.8.0.43.Linux.x86_64-gnu.cuda-12.8.tar.gz"
  wget --no-verbose ${RELEASE_URL_TRT} -O /tmp/TensorRT.tar
  tar -xf /tmp/TensorRT.tar -C /usr/local/
  mv /usr/local/TensorRT-${TRT_VER} /usr/local/tensorrt
  pip3 install /usr/local/tensorrt/python/tensorrt-*-cp${PARSED_PY_VERSION}-*.whl
  rm -rf /tmp/TensorRT.tar

  cp -rv /usr/local/tensorrt/include/* /usr/include/x86_64-linux-gnu/
  cp -rv /usr/local/tensorrt/lib/* /usr/lib/x86_64-linux-gnu/
}

install_tensorrt_aarch64(){
  if [[ $(apt list --installed | grep libnvinfer) ]]; then apt-get remove --purge -y libnvinfer*; fi
  if [[ $(apt list --installed | grep tensorrt) ]]; then apt-get remove --purge -y tensorrt*; fi
  pip3 uninstall -y tensorrt

  PY_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[0:2])))')
  PARSED_PY_VERSION=$(echo "${PY_VERSION//./}")

  # install tensorrt
  RELEASE_URL_TRT="https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.8.0/tars/TensorRT-10.8.0.43.Linux.aarch64-gnu.cuda-12.8.tar.gz"
  wget --no-verbose ${RELEASE_URL_TRT} -O /tmp/TensorRT.tar
  tar -xf /tmp/TensorRT.tar -C /usr/local/
  mv /usr/local/TensorRT-${TRT_VER} /usr/local/tensorrt
  pip3 install /usr/local/tensorrt/python/tensorrt-*-cp${PARSED_PY_VERSION}-*.whl
  rm -rf /tmp/TensorRT.tar

  cp -rv /usr/local/tensorrt/include/* /usr/include/aarch64-linux-gnu/
  cp -rv /usr/local/tensorrt/lib/* /usr/lib/aarch64-linux-gnu/
}


install_tensorrt_soc(){
  if [[ ${USE_DEB_PKG} -eq 1 ]]; then
    # install from DEB
    if [[ $(apt list --installed | grep libnvinfer) ]]; then apt-get remove --purge -y libnvinfer*; fi
    if [[ $(apt list --installed | grep tensorrt) ]]; then apt-get remove --purge -y tensorrt*; fi
    apt autoremove -y
    pip3 uninstall -y tensorrt
  
    export TRT_DEB_URL=http://cuda-repo.nvidia.com/release-candidates/Libraries/TensorRT/v10.13/${TRT_VER}-89f5cfff/${CUDA_VER}-r570/d7l-aarch64/deb/
    install_deb_pkg libnvinfer${TRT_MAJOR_VER}_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg libnvinfer-headers-dev_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg libnvinfer-dev_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg libnvinfer-headers-plugin-dev_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg libnvinfer-lean${TRT_MAJOR_VER}_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg libnvinfer-lean-dev_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg libnvinfer-dispatch${TRT_MAJOR_VER}_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg libnvinfer-dispatch-dev_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg libnvinfer-plugin${TRT_MAJOR_VER}_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg libnvinfer-plugin-dev_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg libnvinfer-vc-plugin${TRT_MAJOR_VER}_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg libnvinfer-vc-plugin-dev_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg libnvonnxparsers${TRT_MAJOR_VER}_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg libnvonnxparsers-dev_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg python3-libnvinfer_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg python3-libnvinfer-lean_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg python3-libnvinfer-dispatch_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg python3-libnvinfer-dev_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg libnvinfer-bin_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    ln -sf /usr/src/tensorrt/bin/trtexec /usr/bin/trtexec
    unset -f install_deb_pkg
  else
    # install from tar
    if [[ $(apt list --installed | grep libnvinfer) ]]; then apt-get remove --purge -y libnvinfer*; fi
    if [[ $(apt list --installed | grep tensorrt) ]]; then apt-get remove --purge -y tensorrt*; fi
    pip3 uninstall -y tensorrt

    PY_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[0:2])))')
    PARSED_PY_VERSION=$(echo "${PY_VERSION//./}")

    RELEASE_URL_TRT="http://cuda-repo/release-candidates/Libraries/TensorRT/private/v10.13/${TRT_VER}-a473f855/${CUDA_VER}-r570/d7l-aarch64/tar/TensorRT-${TRT_VER}.d7l.aarch64-gnu.cuda-${CUDA_VER}.tar.gz"
    wget --no-verbose ${RELEASE_URL_TRT} -O /tmp/TensorRT.tar.gz
    tar -xf /tmp/TensorRT.tar.gz -C /usr/local/
    mv /usr/local/TensorRT-${TRT_VER} /usr/local/tensorrt
    pip3 install /usr/local/tensorrt/python/tensorrt-*-cp${PARSED_PY_VERSION}-*.whl
    rm -rf /tmp/TensorRT.tar

    cp -rv /usr/local/tensorrt/include/* /usr/include/aarch64-linux-gnu/
    cp -rv /usr/local/tensorrt/lib/* /usr/lib/aarch64-linux-gnu/
    cp -rv /usr/local/tensorrt/bin/* /usr/bin/
  fi
}

case $BUILD_CONTEXT in
  x86_64)
    if [ "$USE_NIGHTLY" = "1" ]; then
      install_tensorrt_x86_64_nightly
    else
      install_tensorrt_x86_64
    fi
    ;;
  aarch64-Grace)
    install_tensorrt_aarch64
    ;;
  aarch64-SoC)
    install_tensorrt_soc
    ;;
  *)
    echo "Supported BUILD_CONTEXT are only x86_64, aarch64-Grace, and aarch64-SoC."
    exit 1
    ;;
esac
