# BEVFormer Benchmark

This benchmark performs 3D object detection using the [BEVFormer-Tiny](http://arxiv.org/abs/2203.17270) network and the nuScenes dataset.

:warning: **IMPORTANT**: Please use [closed/NVIDIA](closed/NVIDIA) as the working directory when
running the below commands. :warning:

## Dataset

### Downloading / obtaining the dataset

The dataset used for this benchmark is [nuScenes](https://nuscenes.org/nuscenes). Please manually download the dataset from [MLCommons nuScenes](https://nuscenes.mlcommons.org/) after EULA agreement. Please put the dataset under `$MLPERF_SCRATCH_PATH/data/nuScenes/`.

### Preprocessing the dataset for usage

To process the input images to INT8 linear format for img category and FP16 for other categories, please run `BENCHMARKS=bevformer make preprocess_data`. The preprocessed data will be saved to `$MLPERF_SCRATCH_PATH/preprocessed_data/nuscenes/val/int8/`.

## Model

### Downloading / obtaining the model

The ONNX model *bevformer_tiny.onnx* can be downloaded from the [MLCommons bevformer](https://nuscenes.mlcommons.org/) after EULA agreement. Please put the model under `$MLPERF_SCRATCH_PATH/models/bevformer_tiny`

### Optimizations

#### Plugins

The following TensorRT plugins are used to optimize BEVFormer benchmark:
- `$MLPERF_SCRATCH_PATH/plugins/libbev_plugins.so`: fused and optimized implementation for MultiScaleDeformableAttention, Rotate, Select, FusedCanbus and FusedLidar2img layers.

#### Lower Precision

To further optimize performance, with minimal impact on classification accuracy, we run the computations in INT8 precision for backbone and mixed precision for transformer.

#### Redundant layer transformation and removal

As multiple `ScatterND`, `Gather`, `Transpose` layers have no affect to the final results, we removed some of them. Also, we fused some `MatMul` horizontally to same the memory traffic and removed some mathematically inversible layers like sigmoid and inverse_sigmoid. We further fused several subgraph into plugins to reduce the number of launched kernels and IO.

### Calibration

BEVFormer INT8 is calibrated on a MLCommons provided calibration set - images from randomly selected scenes in nuScenes training set. The indices of this subset can be found at
`closed/NVIDIA/code/bevformer/tensorrt/calibration/calib_order.txt`. We utilize [TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) to help us insert `QuantizeLinear` and `DequantizeLinear` layers and calculate the scales. 

(OPTIONAL) To obtain these scales, please follow below instructions.
```bash
# On any x86 machine with gpu
cd /mlperf-automotive/closed/NVIDIA/docker/calibration
# you only need to build the docker once
docker build --network=host -t mlcommons_bevformer_calibration -f Dockerfile .
# launch the docker
docker run --gpus all -it --network=host --rm \
  -v $MLPERF_SCRATCH_PATH:/data/ \
  -v $Path_to_mlperf-automotive/closed/NVIDIA/:/work/ \
  -v $TENSORRT_ROOT:/tensorrt/ \
  mlcommons_bevformer_calibration

# inside the docker
cd /work/code/bevformer/tensorrt/calibration
export MLPERF_SCRATCH_PATH=/data
python3 prepare_calibration_data.py --output_dir=/work/build/calibration/
python3 prepare_scales.py
```
After above commands, you will see `scale_cache.json` inside `closed/NVIDIA/build/calibration`.

## Instructions for Audits

### Run Inference through LoadGen

Run the following commands from within the container to run inference through LoadGen:

```
make run RUN_ARGS="--benchmarks=bevformer --scenarios=<SCENARIO> --test_mode=PerformanceOnly"
make run RUN_ARGS="--benchmarks=bevformer --scenarios=<SCENARIO> --test_mode=AccuracyOnly"
```

The performance and the accuracy results will be printed to stdout, and the LoadGen logs can be found in `build/logs`.
