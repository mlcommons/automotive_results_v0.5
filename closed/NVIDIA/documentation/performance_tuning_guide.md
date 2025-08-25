# NVIDIA MLPerf Automotive System Under Test (SUT) Performance Tuning Guide
This guide is meant to provide details on how to fix issues involving bad performance, "invalid results", or potential hard crashes after following the "Prepping our repo for your machine" section of the MLPerf Automotive Tutorial (this is the same as the closed/NVIDIA/README.md stored in the repo).

After you add your system description to system_list.py and add configs for your desired benchmarks, it is possible there might be issues causing you to not achieve the best performance possible. Before diagnosing any possible errors, **please go through each of these sections in order, as some pertain to adhering to MLPerf Automotive rules**.

**IMPORTANT:** Make sure your performance tuning changes (i.e. any change made following steps on this document) are done in `configs/[BENCHMARK]/[SCENARIO]/__init__.py` files. Note that all files in the `measurements/` directory are automatically generated from the files in the `configs/` directory at runtime, so any manual changes made in `measurements/` will not take effect.

### Using NSYS to inspect performance
NSYS profiles are useful in gaining insight as to where opportunities for performance optimizations lie. Note that some C++ harness implementations require enable NVTX marker debug flag in build phase to add useful, but performance impacting NVTX markers.
```
nsys profile --force-overwrite=true --gpu-metrics-devices=all --soc-metrics=true -t "cuda,cudla,osrt,nvtx,tegra-accelerators" --cuda-memory-usage=true --accelerator-trace=tegra-accelerators -y 270 –d 30 -o profile.nsys-rep ​make run_harness RUN_ARGS="--benchmarks=bevformer –scenarios=SingleStream​"
```

### Different system configurations that use the same GPU configuration
Sometimes, it may be the case that submitters will have 2 different submission systems with the same GPU configuration, but differing hardware configurations elsewhere, such as a different CPU, memory size, etc. These are counted as separate systems, and you should have definitions for these which specify the CPUs and memory sizes, as well as different system IDs. This also means you will need separate benchmark configurations for each unique system. See the main README.md for instructions on how to add a system. If you are using the automated script, you will need to run it once on each system.

### System configuration tips
For systems with passively cooled GPUs, the cooling system in hardware plays an important role in performance. To check for thermal throttling, run `tegrastats` to monitor the GPU temperature while the harness is running. Ideally, the GPU temperature should saturate or stabilize to a reasonable temperature, such as 65C. If the temperature is erratic or spiking, or if the GPU clock frequencies are unstable, you may need to improve your system's cooling solution.

### Engine build failures
Sometimes, when simply copy-pasting configuration blocks, you can run into hard crashes in the `generate_engines` (engine build) step. While there is no sure-fire way to determine the exact error, you can go through the following possibilities to try and resolve the problem:

1. Try disabling CUDA graphs.
2. Make sure all the required plugins are available from designated path.
3. Check if batch size is set properly.

### Using expected runtime to recognize issues
The [MLPerf Automotive rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc) requires each submission to meet certain requirements to become a valid submission. One of these is the requirements is the runtime of the benchmark test. Below, we summarize some mathematical formulas to determine approximate benchmark runtimes.

**SingleStream:**

Default test runtime:

```
(min_duration / single_stream_expected_latency_ns * actual_latency)
```
Recommended test runtime:

```
max((min_duration / single_stream_expected_latency_ns * actual_latency), (min_query_count * actual_latency))
```
Where:

- `min_duration`: 600 seconds by default
- `single_stream_expected_latency_ns`: set by the benchmark's BenchmarkConfiguration
- `min_query_count`: 6,636 by default

The typical runtime for SingleStream scenario should be about 600 seconds (60 seconds with `--test_run`), unless on a system with a very long latency per sample. In this case, runtime will be much longer.

**ConstantStream:**

Default test runtime:

```
max(min_duration, min_query_count / constant_stream_target_qps)
```

Where:

- `min_duration`: 600 seconds by default, 60 seconds with `--test_run`
- `constant_stream_target_qps`: set by the benchmark's BenchmarkConfiguration
- `min_query_count`: 100,000 by default

Depending on the performance and latency behavior of the system, the runtime can keep increasing in "Default" mode because the Early Stopping mechanism decides that the current number of samples do not have enough statistical guarantees.

**Early stopping**: *min_query_count* is not a requirement for runtime if it becomes prohibitively long. *Early Stopping* allows for systems to process a smaller number of queries during their runs than previously allowed. After we have run for *min_duration*, the system can ask whether the overall processed number of queries already provide statistical guarantees of the reported performance number. If that is not the case, the system will suggest an extended number of queries to run. The system should stop at any case if the current processed number of queries reaches *min_query_count* as is unlikely that running any further would improve the chances of having a successful run (or actual convergence at all, meaning unbounded runtime).

### Fixing INVALID results
An INVALID result occurs when the harness finishes running successfully, but does not fulfill all of the runtime requirements to be considered valid for an official submission.

**SingleStream:**
The most common reason for INVALID results in SingleStream scenario is that the actual latency of the system per sample is much lower than the specified SingleStream expected latency. Therefore, simply lower `single_stream_expected_latency_ns` to match the actual 90th percentile latency reported by LoadGen.

**ConstantStream:**
The most common reason for INVALID results in ConstantStream scenario is that the actual latency of the system per sample is much higher to meet the constant stream target QPS. In this case, it is required to lower the `constant_stream_target_qps` and acknowledge this to the MLPerf Autmomotive committee. 

