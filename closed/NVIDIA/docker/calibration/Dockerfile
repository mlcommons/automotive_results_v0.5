FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

RUN chmod -R 777 /tmp
RUN apt update
RUN apt install libgl1-mesa-glx libsm6 libxext6  -y
RUN apt install python3 python3-pip wget -y
RUN apt install cmake git -y

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install debugpy Cython setuptools==65 ninja
RUN python3 -m pip install onnx==1.17.0 onnx_graphsurgeon onnxsim nvidia-modelopt==0.31.0 --extra-index-url https://pypi.ngc.nvidia.com
RUN pip install onnxconverter_common==1.14.0 onnxruntime-gpu==1.22.0
