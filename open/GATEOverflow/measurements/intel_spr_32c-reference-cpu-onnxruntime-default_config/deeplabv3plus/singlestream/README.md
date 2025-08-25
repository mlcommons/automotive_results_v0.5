*Check [MLC MLPerf docs](https://docs.mlcommons.org/automotive) for more details.*

## Host platform

* OS version: Linux-6.11.0-25-generic-x86_64-with-glibc2.39
* CPU version: x86_64
* Python version: 3.12.3 (main, Jun 18 2025, 17:59:45) [GCC 13.3.0]
* MLC version: unknown

## MLC Run Command

See [MLC installation guide](https://docs.mlcommons.org/mlcflow/install/).

```bash
pip install -U mlcflow

mlc rm cache -f

mlc pull repo gateoverflow@mlperf-automations --checkout=23b3d356b5e87286514538ed91c0fbd5fa252a33


```
*Note that if you want to use the [latest automation recipes](https://docs.mlcommons.org/inference) for MLPerf,
 you should simply reload gateoverflow@mlperf-automations without checkout and clean MLC cache as follows:*

```bash
mlc rm repo gateoverflow@mlperf-automations
mlc pull repo gateoverflow@mlperf-automations
mlc rm cache -f

```

## Results

Platform: intel_spr_32c-reference-cpu-onnxruntime-default_config

Model Precision: fp32

### Accuracy Results 
`mIOU`: `0.92436`, Required accuracy for closed division `>= 0.92343`

### Performance Results 
`90th percentile latency (ns)`: `2481297472.0`
