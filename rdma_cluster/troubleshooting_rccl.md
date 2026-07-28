# Issue Report: vLLM Tensor Parallelism over RDMA on AMD Strix Halo

> **✅ RESOLVED (Feb 2, 2026)**
> This issue is **SOLVED**. The root cause was indeed missing `gfx1151` support in the upstream RCCL library.
>
> I have patched and built a custom version of RCCL with native `gfx1151` support. This patched library is **now included** in the toolbox container provided by this repository (`kyuz0/vllm-therock-gfx1151`).
>
> See the [RDMA Cluster Setup Guide](setup_guide.md) for instructions on how to run the cluster using the fixed container.

## TL;DR
I am attempting to run vLLM with Tensor Parallelism across two AMD Strix Halo (Ryzen AI MAX+ 395) nodes connected directly via Intel E810-CQDA1 RDMA (RoCE v2).

- **Current Status:** RDMA communication is verified (low latency ~5us). Ray cluster is operational and can allocate tensors on both nodes.
- **Blocker:** vLLM fails with `HIP error: invalid kernel file` when initializing the distributed environment.
- **Suspected Cause:** Possible missing support for `gfx1151` in the RCCL library included with the ROCm nightly build.
- **Goal:** Solicit troubleshooting advice or confirmation if `gfx1151` support is indeed missing/required in RCCL.

## Table of Contents
1. [Context & Goal](#1-context--goal)
2. [System Setup](#2-system-setup)
3. [Verification Steps](#3-verification-steps)
    - [3.1 RDMA/RoCE Link Test](#31-rdmaroce-link-test)
    - [3.2 Ray Cluster & Torch Verification](#32-ray-cluster--torch-verification)
4. [The Issue: Invalid Kernel File](#4-the-issue-invalid-kernel-file)
    - [4.1 Command & Configuration](#41-command--configuration)
    - [4.2 Error Logs](#42-error-logs)
    - [4.3 Hypothesis: RCCL Support for gfx1151](#43-hypothesis-rccl-support-for-gfx1151)
5. [Request for Help](#5-request-for-help)

## 1. Context & Goal
The objective is to enable **Tensor Parallelism** with vLLM distributed across two machines.
*   **Hardware:** Two AMD Strix Halo dev kits (Ryzen AI MAX+ 395).
*   **Interconnect:** Direct 100GbE connection using Intel E810 cards configured for RoCE v2.
*   **Software Stack:** Fedora 43, ROCm v7.12 (Nightlies), vLLM (Source build), Ray.

I have established a stable low-latency RDMA link and a functional Ray cluster over that link. However, the actual vLLM engine initialization crashes.

## 2. System Setup

### Hardware Configuration
*   **Nodes:** 2x AMD Strix Halo APUs (Framework Laptop / Dev Kit).
*   **CPU/GPU:** AMD Ryzen AI MAX+ 395 w/ Radeon 8060S (128GB Unified Memory via GTT).
*   **Interconnect:** Direct connection via Intel Ethernet Controller E810-CQDA1.
*   **Protocol:** RoCE v2 (RDMA over Converged Ethernet).

### Host Software (Fedora)
| Node | Hostname | Kernel | OS | IP (RDMA Interface) |
| :--- | :--- | :--- | :--- | :--- |
| **Node 1** | `frmwk-dsk` | `6.18.5-200.fc43.x86_64` | Fedora Linux 43 | `192.168.100.1/30` |
| **Node 2** | `fw2` | `6.18.6-200.fc43.x86_64` | Fedora Linux 43 | `192.168.100.2/30` |

### Container Environment (Toolbox)
Both nodes run identical Toolboxes based on **Fedora 43** with **TheRock ROCm Nightlies**.
*   **Container Runtime:** Podman 5.7.1 / Toolbox 0.3.
*   **RDMA Device:** `rocep194s0` (State: `PORT_ACTIVE`, Link: `Ethernet`).
*   **ROCm Version:** `7.12.0` (TheRock Nightly `20260131`).
*   **Python:** 3.13.11
*   **Compiler:** GCC 15.2.1

**Key Packages:**
| Package | Version | Source |
| :--- | :--- | :--- |
| **vLLM** | `0.16.0rc1.dev55+g8980001c9.d20260131.rocm712` | Source Build (Nightly) |
| **PyTorch** | `2.11.0a0+rocm7.12.0a20260131` | TheRock Nightly |
| **Triton** | `3.6.0+git447f725f.rocm7.12.0a20260131` | TheRock Nightly |
| **RCCL** | `2.27.7` (Built-in) | TheRock Nightly |

### Toolbox Creation Command
The environment is created using `toolbox` (wrapping Podman) with specific flags to expose the Infiniband devices and unlock memory limits for RDMA:

```bash
toolbox create vllm \
  --image docker.io/kyuz0/vllm-therock-gfx1151:latest \
  -- \
  --device /dev/dri \
  --device /dev/kfd \
  --device /dev/infiniband \
  --group-add keep-groups \
  --ulimit memlock=-1 \
  --security-opt seccomp=unconfined
```

> `--group-add keep-groups` passes your host `render`/`video`/`rdma` groups through. Named
> `--group-add video --group-add render --group-add rdma` does **not** work under rootless
> podman and aborts the container if the image lacks the group — see
> [GPU Device Permissions](../README.md#61-gpu-device-permissions).

The following Dockerfile is used for the toolbox: https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes/blob/main/Dockerfile


## 3. Verification Steps

### 3.1. RDMA/RoCE Link Test

Running the `compare_eth_vs_rdma.sh` shows indeed that this is working fine:

```bash
kyuz0@frmwk-dsk:~$ ./compare_eth_vs_rdma.sh 

=== Network Comparison ===

Path                 Latency      Bandwidth   
------------------------------------------------
Ethernet (1G LAN)    0.074 ms     0.94 Gbps   
Ethernet (RoCE NIC)  0.068 ms     55.70 Gbps  
RDMA (RoCE)          5.23 us      50.64 Gbps  
```

### 3.2. Ray Cluster & Torch Verification

Before attempting distributed inference with vLLM, we established a Ray cluster to verify three things:
1.  Control plane communication over the RDMA interface (`192.168.100.x`).
2.  Hardware visibility (GPU) on both nodes under Ray's management.
3.  Basic PyTorch tensor allocation capabilities on the Strix Halo APUs.

#### Head Node Configuration (Node 1)
We explicitly bind Ray's internal communication (Gloo) and the NCCL socket discovery to the high-speed interface. Crucially, we set `RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1` to prevent Ray from interfering with the APU visibility, which is a known issue on ROCm 6.x+ with integrated graphics.

```bash
ray stop --force

# Detect RDMA interface
export RDMA_IFACE=$(ip -o addr show to 192.168.100.0/24 | awk '{print $2}' | head -n1)

# Bind Ray/Torch communication
export GLOO_SOCKET_IFNAME=$RDMA_IFACE
export NCCL_SOCKET_IFNAME=$RDMA_IFACE

# Prevent Ray from masking the APU (Strix Halo Requirement)
export RAY_DISABLE_METRICS=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1

# Start Head
ray start --head --port=6379 --node-ip-address=192.168.100.1 --num-gpus=1 --num-cpus=8 --disable-usage-stats
```

#### Worker Node Configuration (Node 2)
The worker joins using the same environment settings.

```bash
ray stop --force

export RDMA_IFACE=$(ip -o addr show to 192.168.100.0/24 | awk '{print $2}' | head -n1)
export GLOO_SOCKET_IFNAME=$RDMA_IFACE
export NCCL_SOCKET_IFNAME=$RDMA_IFACE
export RAY_DISABLE_METRICS=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1

ray start --address='192.168.100.1:6379' --num-gpus=1 --num-cpus=8 --disable-usage-stats
```

#### Cluster Status
Running `ray status` confirms both nodes are active, providing a total of 2 GPUs (1 per APU) to the pool.

```text
======== Autoscaler status: 2026-01-31 14:20:05.449661 ========
Node status
---------------------------------------------------------------
Active:
 1 node_5a3b1f77fbfa9166383761707a12bb40f0a03d4f635e26acf703b0cc
 1 node_709eb6806677f4171ba13b3fc6aee20da41d68f3ef22acbce6fce967
Pending:
 (no pending nodes)
Recent failures:
 (no failures)

Resources
---------------------------------------------------------------
Total Usage:
 0.0/16.0 CPU
 0.0/2.0 GPU
 0.0/1.0 TPU
 0B/167.19GiB memory
 0B/71.65GiB object_store_memory

From request_resources:
 (none)
Pending Demands:
 (no resource demands)

```

### 3.4. Functional Verification Script
We ran a python script to submit remote tasks to both nodes, ensuring they can initialize the ROCm driver and allocate tensors.

**Script (`verify_cluster.py`):**
```python
import ray
import torch
import socket
import os

print("Connecting to Ray Cluster...")
ray.init(address="192.168.100.1:6379")

@ray.remote(num_gpus=1)
def check_node_gpu():
    host = socket.gethostname()
    ip = socket.gethostbyname(host)
    try:
        gpu_name = torch.cuda.get_device_name(0)
        # Perform a tiny allocation to prove the driver is accepting commands
        t = torch.tensor([1.0], device="cuda")
        return f"SUCCESS: Node {host} ({ip}) - {gpu_name} - Tensor: {t}"
    except Exception as e:
        return f"FAILURE: Node {host} ({ip}) - Error: {e}"

print("Submitting tasks to both nodes...")
results = ray.get([check_node_gpu.remote() for _ in range(2)])

print("\n--- Cluster Status ---")
for res in results:
    print(r"  " + res)
```

**Output:**
```text
Connecting to Ray Cluster...
2026-01-31 13:51:16,452	INFO worker.py:1821 -- Connecting to existing Ray cluster at address: 192.168.100.1:6379...
2026-01-31 13:51:16,461	INFO worker.py:2007 -- Connected to Ray cluster.
...
--- Cluster Status ---
  SUCCESS: Node toolbx (192.168.1.128) - Radeon 8060S Graphics - Tensor: tensor([1.], device='cuda:0')
  SUCCESS: Node toolbx (192.168.1.127) - Radeon 8060S Graphics - Tensor: tensor([1.], device='cuda:0')
```

## 4. The Issue: Invalid Kernel File

Now that the cluster is setup and demonstrated to work, I try to run vLLM with tensor-parallelism like this:

```bash
(venv) ⬢ [kyuz0@toolbx ~]$ # 1. Launcher Environment (Including RCCL/RDMA vars)
export VLLM_HOST_IP=192.168.100.1
export NCCL_SOCKET_IFNAME=$(ip -o addr show to 192.168.100.0/24 | awk '{print $2}' | head -n1)
export NCCL_IB_GID_INDEX=1
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=0
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1

# 2. Launch
vllm serve facebook/opt-125m \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --trust-remote-code \
  --enforce-eager
```

### 4.2. Error Logs

This results in an `HIP error: invalid kernel file` immediately upon engine initialization:

```text
(APIServer pid=17743) INFO 01-31 14:20:54 [utils.py:346] 
(APIServer pid=17743) INFO 01-31 14:20:54 [utils.py:346]        █     █     █▄   ▄█
(APIServer pid=17743) INFO 01-31 14:20:54 [utils.py:346]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.16.0rc1.dev55+g8980001c9.d20260131
(APIServer pid=17743) INFO 01-31 14:20:54 [utils.py:346]   █▄█▀ █     █     █     █  model   facebook/opt-125m
(APIServer pid=17743) INFO 01-31 14:20:54 [utils.py:346]    ▀▀  ▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀
(APIServer pid=17743) INFO 01-31 14:20:54 [utils.py:346] 
(APIServer pid=17743) INFO 01-31 14:20:54 [utils.py:282] non-default args: {'model_tag': 'facebook/opt-125m', 'api_server_count': 1, 'model': 'facebook/opt-125m', 'trust_remote_code': True, 'enforce_eager': True, 'distributed_executor_backend': 'ray', 'tensor_parallel_size': 2}
(APIServer pid=17743) The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.
(APIServer pid=17743) The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.
(APIServer pid=17743) INFO 01-31 14:20:55 [model.py:541] Resolved architecture: OPTForCausalLM
(APIServer pid=17743) INFO 01-31 14:20:55 [model.py:1559] Using max model len 2048
(APIServer pid=17743) INFO 01-31 14:20:55 [scheduler.py:226] Chunked prefill is enabled with max_num_batched_tokens=8192.
(APIServer pid=17743) WARNING 01-31 14:20:55 [vllm.py:651] Async scheduling will be disabled because it is not supported with the `ray` distributed executor backend (only `mp`, `uni`, and `external_launcher` are supported).
(APIServer pid=17743) INFO 01-31 14:20:55 [vllm.py:669] Asynchronous scheduling is disabled.
(APIServer pid=17743) WARNING 01-31 14:20:55 [vllm.py:707] Enforce eager set, overriding optimization level to -O0
(APIServer pid=17743) INFO 01-31 14:20:55 [vllm.py:807] Cudagraph is disabled under eager mode
(EngineCore_DP0 pid=17822) INFO 01-31 14:20:59 [core.py:96] Initializing a V1 LLM engine (v0.16.0rc1.dev55+g8980001c9.d20260131) with config: model='facebook/opt-125m', speculative_config=None, tokenizer='facebook/opt-125m', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.float16, max_seq_len=2048, download_dir=None, load_format=auto, tensor_parallel_size=2, pipeline_parallel_size=1, data_parallel_size=1, disable_custom_all_reduce=True, quantization=None, enforce_eager=True, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=0, served_model_name=facebook/opt-125m, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'level': None, 'mode': <CompilationMode.NONE: 0>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['all', '+sparse_attn_indexer'], 'splitting_ops': [], 'compile_mm_encoder': False, 'compile_sizes': [], 'compile_ranges_split_points': [8192], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.NONE: 0>, 'cudagraph_num_of_warmups': 0, 'cudagraph_capture_sizes': [], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': False, 'fuse_act_quant': False, 'fuse_attn_quant': False, 'eliminate_noops': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False, 'fuse_act_padding': False}, 'max_cudagraph_capture_size': 0, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'static_all_moe_layers': []}
(EngineCore_DP0 pid=17822) 2026-01-31 14:20:59,483	INFO worker.py:1821 -- Connecting to existing Ray cluster at address: 192.168.100.1:6379...
(EngineCore_DP0 pid=17822) 2026-01-31 14:20:59,492	INFO worker.py:2007 -- Connected to Ray cluster.
(EngineCore_DP0 pid=17822) /opt/venv/lib64/python3.13/site-packages/ray/_private/worker.py:2046: FutureWarning: Tip: In future versions of Ray, Ray will no longer override accelerator visible devices env var if num_gpus=0 or num_gpus=None (default). To enable this behavior and turn off this error message, set RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
(EngineCore_DP0 pid=17822)   warnings.warn(
(EngineCore_DP0 pid=17822) INFO 01-31 14:20:59 [ray_utils.py:402] No current placement group found. Creating a new placement group.
(EngineCore_DP0 pid=17822) WARNING 01-31 14:20:59 [ray_utils.py:213] tensor_parallel_size=2 is bigger than a reserved number of GPUs (1 GPUs) in a node 5a3b1f77fbfa9166383761707a12bb40f0a03d4f635e26acf703b0cc. Tensor parallel workers can be spread out to 2+ nodes which can degrade the performance unless you have fast interconnect across nodes, like Infiniband. To resolve this issue, make sure you have more than 2 GPUs available at each node.
(EngineCore_DP0 pid=17822) WARNING 01-31 14:20:59 [ray_utils.py:213] tensor_parallel_size=2 is bigger than a reserved number of GPUs (1 GPUs) in a node 709eb6806677f4171ba13b3fc6aee20da41d68f3ef22acbce6fce967. Tensor parallel workers can be spread out to 2+ nodes which can degrade the performance unless you have fast interconnect across nodes, like Infiniband. To resolve this issue, make sure you have more than 2 GPUs available at each node.
(EngineCore_DP0 pid=17822) INFO 01-31 14:21:02 [ray_env.py:66] RAY_NON_CARRY_OVER_ENV_VARS from config: set()
(EngineCore_DP0 pid=17822) INFO 01-31 14:21:02 [ray_env.py:69] Copying the following environment variables to workers: ['VLLM_ROCM_USE_AITER', 'VLLM_WORKER_MULTIPROC_METHOD', 'MAX_JOBS', 'LD_LIBRARY_PATH', 'VLLM_USE_TRITON_AWQ', 'VLLM_TARGET_DEVICE', 'VLLM_ROCM_USE_AITER_MOE']
(EngineCore_DP0 pid=17822) INFO 01-31 14:21:02 [ray_env.py:74] If certain env vars should NOT be copied, add them to /home/kyuz0/.config/vllm/ray_non_carry_over_env_vars.json file
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) WARNING 01-31 14:21:03 [worker_base.py:301] Missing `shared_worker_lock` argument from executor. This argument is needed for mm_processor_cache_type='shm'.
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) INFO 01-31 14:21:03 [parallel_state.py:1234] world_size=2 rank=0 local_rank=0 distributed_init_method=tcp://192.168.100.1:49483 backend=nccl
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) INFO 01-31 14:21:03 [pynccl.py:111] vLLM is using nccl==2.27.7
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946] EngineCore failed to start.
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946] Traceback (most recent call last):
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/engine/core.py", line 937, in run_engine_core
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/engine/core.py", line 691, in __init__
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     super().__init__(
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     ~~~~~~~~~~~~~~~~^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]         vllm_config,
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]         ^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     ...<3 lines>...
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]         internal_dp_balancing,
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]         ^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     )
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     ^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/engine/core.py", line 105, in __init__
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     self.model_executor = executor_class(vllm_config)
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]                           ~~~~~~~~~~~~~~^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/executor/abstract.py", line 101, in __init__
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     self._init_executor()
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     ~~~~~~~~~~~~~~~~~~~^^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/executor/ray_executor.py", line 97, in _init_executor
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     self._init_workers_ray(placement_group)
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/executor/ray_executor.py", line 366, in _init_workers_ray
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     self.collective_rpc("init_device")
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/executor/ray_executor.py", line 489, in collective_rpc
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     return ray.get(ray_worker_outputs, timeout=timeout)
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]            ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]   File "/opt/venv/lib64/python3.13/site-packages/ray/_private/auto_init_hook.py", line 22, in auto_init_wrapper
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     return fn(*args, **kwargs)
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]   File "/opt/venv/lib64/python3.13/site-packages/ray/_private/client_mode_hook.py", line 104, in wrapper
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     return func(*args, **kwargs)
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]   File "/opt/venv/lib64/python3.13/site-packages/ray/_private/worker.py", line 2967, in get
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     values, debugger_breakpoint = worker.get_objects(
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]                                   ~~~~~~~~~~~~~~~~~~^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]         object_refs, timeout=timeout, _tensor_transport=_tensor_transport
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     )
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     ^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]   File "/opt/venv/lib64/python3.13/site-packages/ray/_private/worker.py", line 1015, in get_objects
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     raise value.as_instanceof_cause()
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946] ray.exceptions.RayTaskError(AcceleratorError): ray::RayWorkerWrapper.execute_method() (pid=17931, ip=192.168.100.1, actor_id=e2ed7f9c6fbd6274f30c736001000000, repr=<vllm.v1.executor.ray_utils.RayWorkerWrapper object at 0x7ff79ca5c590>)
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/worker/worker_base.py", line 345, in execute_method
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     raise e
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/worker/worker_base.py", line 334, in execute_method
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     return run_method(self, method, args, kwargs)
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/serial_utils.py", line 459, in run_method
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     return func(*args, **kwargs)
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/worker/worker_base.py", line 326, in init_device
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     self.worker.init_device()  # type: ignore
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     ~~~~~~~~~~~~~~~~~~~~~~~^^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/worker/gpu_worker.py", line 221, in init_device
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     init_worker_distributed_environment(
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]         self.vllm_config,
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]         ^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     ...<3 lines>...
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]         current_platform.dist_backend,
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     )
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     ^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/worker/gpu_worker.py", line 959, in init_worker_distributed_environment
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     ensure_model_parallel_initialized(
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]         parallel_config.tensor_parallel_size,
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     ...<2 lines>...
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]         parallel_config.decode_context_parallel_size,
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     )
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     ^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]   File "/opt/venv/lib64/python3.13/site-packages/vllm/distributed/parallel_state.py", line 1472, in ensure_model_parallel_initialized
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     initialize_model_parallel(
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     ~~~~~~~~~~~~~~~~~~~~~~~~~^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]         tensor_model_parallel_size,
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     ...<3 lines>...
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]         backend,
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]         ^^^^^^^^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     )
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     ^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]   File "/opt/venv/lib64/python3.13/site-packages/vllm/distributed/parallel_state.py", line 1369, in initialize_model_parallel
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     _TP = init_model_parallel_group(
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]         group_ranks,
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     ...<3 lines>...
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]         group_name="tp",
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     )
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]   File "/opt/venv/lib64/python3.13/site-packages/vllm/distributed/parallel_state.py", line 1089, in init_model_parallel_group
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     return GroupCoordinator(
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]         group_ranks=group_ranks,
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     ...<4 lines>...
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]         group_name=group_name,
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     )
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]   File "/opt/venv/lib64/python3.13/site-packages/vllm/distributed/parallel_state.py", line 362, in __init__
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     self.device_communicator = device_comm_cls(
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]                                ~~~~~~~~~~~~~~~^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]         cpu_group=self.cpu_group,
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]         ^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     ...<2 lines>...
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]         unique_name=self.unique_name,
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     )
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     ^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]   File "/opt/venv/lib64/python3.13/site-packages/vllm/distributed/device_communicators/cuda_communicator.py", line 58, in __init__
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     self.pynccl_comm = PyNcclCommunicator(
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]                        ~~~~~~~~~~~~~~~~~~^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]         group=self.cpu_group,
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]         ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]         device=self.device,
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]         ^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     )
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     ^
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]   File "/opt/venv/lib64/python3.13/site-packages/vllm/distributed/device_communicators/pynccl.py", line 145, in __init__
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946]     data = torch.zeros(1, device=device)
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946] torch.AcceleratorError: HIP error: invalid kernel file
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946] Search for `hipErrorInvalidKernelFile' in https://rocm.docs.amd.com/projects/HIP/en/latest/index.html for more information.
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946] HIP kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946] For debugging consider passing AMD_SERIALIZE_KERNEL=3
(EngineCore_DP0 pid=17822) ERROR 01-31 14:21:03 [core.py:946] Device-side assertion tracking was not enabled by user.
(EngineCore_DP0 pid=17822) Process EngineCore_DP0:
(EngineCore_DP0 pid=17822) Traceback (most recent call last):
(EngineCore_DP0 pid=17822)   File "/usr/lib64/python3.13/multiprocessing/process.py", line 313, in _bootstrap
(EngineCore_DP0 pid=17822)     self.run()
(EngineCore_DP0 pid=17822)     ~~~~~~~~^^
(EngineCore_DP0 pid=17822)   File "/usr/lib64/python3.13/multiprocessing/process.py", line 108, in run
(EngineCore_DP0 pid=17822)     self._target(*self._args, **self._kwargs)
(EngineCore_DP0 pid=17822)     ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/engine/core.py", line 950, in run_engine_core
(EngineCore_DP0 pid=17822)     raise e
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/engine/core.py", line 937, in run_engine_core
(EngineCore_DP0 pid=17822)     engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/engine/core.py", line 691, in __init__
(EngineCore_DP0 pid=17822)     super().__init__(
(EngineCore_DP0 pid=17822)     ~~~~~~~~~~~~~~~~^
(EngineCore_DP0 pid=17822)         vllm_config,
(EngineCore_DP0 pid=17822)         ^^^^^^^^^^^^
(EngineCore_DP0 pid=17822)     ...<3 lines>...
(EngineCore_DP0 pid=17822)         internal_dp_balancing,
(EngineCore_DP0 pid=17822)         ^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822)     )
(EngineCore_DP0 pid=17822)     ^
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/engine/core.py", line 105, in __init__
(EngineCore_DP0 pid=17822)     self.model_executor = executor_class(vllm_config)
(EngineCore_DP0 pid=17822)                           ~~~~~~~~~~~~~~^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/executor/abstract.py", line 101, in __init__
(EngineCore_DP0 pid=17822)     self._init_executor()
(EngineCore_DP0 pid=17822)     ~~~~~~~~~~~~~~~~~~~^^
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/executor/ray_executor.py", line 97, in _init_executor
(EngineCore_DP0 pid=17822)     self._init_workers_ray(placement_group)
(EngineCore_DP0 pid=17822)     ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/executor/ray_executor.py", line 366, in _init_workers_ray
(EngineCore_DP0 pid=17822)     self.collective_rpc("init_device")
(EngineCore_DP0 pid=17822)     ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/executor/ray_executor.py", line 489, in collective_rpc
(EngineCore_DP0 pid=17822)     return ray.get(ray_worker_outputs, timeout=timeout)
(EngineCore_DP0 pid=17822)            ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/ray/_private/auto_init_hook.py", line 22, in auto_init_wrapper
(EngineCore_DP0 pid=17822)     return fn(*args, **kwargs)
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/ray/_private/client_mode_hook.py", line 104, in wrapper
(EngineCore_DP0 pid=17822)     return func(*args, **kwargs)
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/ray/_private/worker.py", line 2967, in get
(EngineCore_DP0 pid=17822)     values, debugger_breakpoint = worker.get_objects(
(EngineCore_DP0 pid=17822)                                   ~~~~~~~~~~~~~~~~~~^
(EngineCore_DP0 pid=17822)         object_refs, timeout=timeout, _tensor_transport=_tensor_transport
(EngineCore_DP0 pid=17822)         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822)     )
(EngineCore_DP0 pid=17822)     ^
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/ray/_private/worker.py", line 1015, in get_objects
(EngineCore_DP0 pid=17822)     raise value.as_instanceof_cause()
(EngineCore_DP0 pid=17822) ray.exceptions.RayTaskError(AcceleratorError): ray::RayWorkerWrapper.execute_method() (pid=17931, ip=192.168.100.1, actor_id=e2ed7f9c6fbd6274f30c736001000000, repr=<vllm.v1.executor.ray_utils.RayWorkerWrapper object at 0x7ff79ca5c590>)
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/worker/worker_base.py", line 345, in execute_method
(EngineCore_DP0 pid=17822)     raise e
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/worker/worker_base.py", line 334, in execute_method
(EngineCore_DP0 pid=17822)     return run_method(self, method, args, kwargs)
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/serial_utils.py", line 459, in run_method
(EngineCore_DP0 pid=17822)     return func(*args, **kwargs)
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/worker/worker_base.py", line 326, in init_device
(EngineCore_DP0 pid=17822)     self.worker.init_device()  # type: ignore
(EngineCore_DP0 pid=17822)     ~~~~~~~~~~~~~~~~~~~~~~~^^
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/worker/gpu_worker.py", line 221, in init_device
(EngineCore_DP0 pid=17822)     init_worker_distributed_environment(
(EngineCore_DP0 pid=17822)     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
(EngineCore_DP0 pid=17822)         self.vllm_config,
(EngineCore_DP0 pid=17822)         ^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822)     ...<3 lines>...
(EngineCore_DP0 pid=17822)         current_platform.dist_backend,
(EngineCore_DP0 pid=17822)         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822)     )
(EngineCore_DP0 pid=17822)     ^
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/worker/gpu_worker.py", line 959, in init_worker_distributed_environment
(EngineCore_DP0 pid=17822)     ensure_model_parallel_initialized(
(EngineCore_DP0 pid=17822)     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
(EngineCore_DP0 pid=17822)         parallel_config.tensor_parallel_size,
(EngineCore_DP0 pid=17822)         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822)     ...<2 lines>...
(EngineCore_DP0 pid=17822)         parallel_config.decode_context_parallel_size,
(EngineCore_DP0 pid=17822)         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822)     )
(EngineCore_DP0 pid=17822)     ^
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/distributed/parallel_state.py", line 1472, in ensure_model_parallel_initialized
(EngineCore_DP0 pid=17822)     initialize_model_parallel(
(EngineCore_DP0 pid=17822)     ~~~~~~~~~~~~~~~~~~~~~~~~~^
(EngineCore_DP0 pid=17822)         tensor_model_parallel_size,
(EngineCore_DP0 pid=17822)         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822)     ...<3 lines>...
(EngineCore_DP0 pid=17822)         backend,
(EngineCore_DP0 pid=17822)         ^^^^^^^^
(EngineCore_DP0 pid=17822)     )
(EngineCore_DP0 pid=17822)     ^
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/distributed/parallel_state.py", line 1369, in initialize_model_parallel
(EngineCore_DP0 pid=17822)     _TP = init_model_parallel_group(
(EngineCore_DP0 pid=17822)         group_ranks,
(EngineCore_DP0 pid=17822)     ...<3 lines>...
(EngineCore_DP0 pid=17822)         group_name="tp",
(EngineCore_DP0 pid=17822)     )
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/distributed/parallel_state.py", line 1089, in init_model_parallel_group
(EngineCore_DP0 pid=17822)     return GroupCoordinator(
(EngineCore_DP0 pid=17822)         group_ranks=group_ranks,
(EngineCore_DP0 pid=17822)     ...<4 lines>...
(EngineCore_DP0 pid=17822)         group_name=group_name,
(EngineCore_DP0 pid=17822)     )
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/distributed/parallel_state.py", line 362, in __init__
(EngineCore_DP0 pid=17822)     self.device_communicator = device_comm_cls(
(EngineCore_DP0 pid=17822)                                ~~~~~~~~~~~~~~~^
(EngineCore_DP0 pid=17822)         cpu_group=self.cpu_group,
(EngineCore_DP0 pid=17822)         ^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822)     ...<2 lines>...
(EngineCore_DP0 pid=17822)         unique_name=self.unique_name,
(EngineCore_DP0 pid=17822)         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822)     )
(EngineCore_DP0 pid=17822)     ^
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/distributed/device_communicators/cuda_communicator.py", line 58, in __init__
(EngineCore_DP0 pid=17822)     self.pynccl_comm = PyNcclCommunicator(
(EngineCore_DP0 pid=17822)                        ~~~~~~~~~~~~~~~~~~^
(EngineCore_DP0 pid=17822)         group=self.cpu_group,
(EngineCore_DP0 pid=17822)         ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822)         device=self.device,
(EngineCore_DP0 pid=17822)         ^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822)     )
(EngineCore_DP0 pid=17822)     ^
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/distributed/device_communicators/pynccl.py", line 145, in __init__
(EngineCore_DP0 pid=17822)     data = torch.zeros(1, device=device)
(EngineCore_DP0 pid=17822) torch.AcceleratorError: HIP error: invalid kernel file
(EngineCore_DP0 pid=17822) Search for `hipErrorInvalidKernelFile' in https://rocm.docs.amd.com/projects/HIP/en/latest/index.html for more information.
(EngineCore_DP0 pid=17822) HIP kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
(EngineCore_DP0 pid=17822) For debugging consider passing AMD_SERIALIZE_KERNEL=3
(EngineCore_DP0 pid=17822) Device-side assertion tracking was not enabled by user.
(EngineCore_DP0 pid=17822) INFO 01-31 14:21:03 [ray_executor.py:120] Shutting down Ray distributed executor. If you see error log from logging.cc regarding SIGTERM received, please ignore because this is the expected termination process in Ray.
(EngineCore_DP0 pid=17822) 2026-01-31 14:21:03,366	ERROR worker.py:433 -- Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): ray::RayWorkerWrapper.execute_method() (pid=9409, ip=192.168.100.2, actor_id=4ec5a2c97a621462a00d303801000000, repr=<vllm.v1.executor.ray_utils.RayWorkerWrapper object at 0x7fc7d70fc590>)
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/worker/worker_base.py", line 345, in execute_method
(EngineCore_DP0 pid=17822)     raise e
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/worker/worker_base.py", line 334, in execute_method
(EngineCore_DP0 pid=17822)     return run_method(self, method, args, kwargs)
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/serial_utils.py", line 459, in run_method
(EngineCore_DP0 pid=17822)     return func(*args, **kwargs)
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/worker/worker_base.py", line 326, in init_device
(EngineCore_DP0 pid=17822)     self.worker.init_device()  # type: ignore
(EngineCore_DP0 pid=17822)     ~~~~~~~~~~~~~~~~~~~~~~~^^
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/worker/gpu_worker.py", line 221, in init_device
(EngineCore_DP0 pid=17822)     init_worker_distributed_environment(
(EngineCore_DP0 pid=17822)     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
(EngineCore_DP0 pid=17822)         self.vllm_config,
(EngineCore_DP0 pid=17822)         ^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822)     ...<3 lines>...
(EngineCore_DP0 pid=17822)         current_platform.dist_backend,
(EngineCore_DP0 pid=17822)         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822)     )
(EngineCore_DP0 pid=17822)     ^
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/worker/gpu_worker.py", line 959, in init_worker_distributed_environment
(EngineCore_DP0 pid=17822)     ensure_model_parallel_initialized(
(EngineCore_DP0 pid=17822)     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
(EngineCore_DP0 pid=17822)         parallel_config.tensor_parallel_size,
(EngineCore_DP0 pid=17822)         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822)     ...<2 lines>...
(EngineCore_DP0 pid=17822)         parallel_config.decode_context_parallel_size,
(EngineCore_DP0 pid=17822)         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822)     )
(EngineCore_DP0 pid=17822)     ^
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/distributed/parallel_state.py", line 1472, in ensure_model_parallel_initialized
(EngineCore_DP0 pid=17822)     initialize_model_parallel(
(EngineCore_DP0 pid=17822)     ~~~~~~~~~~~~~~~~~~~~~~~~~^
(EngineCore_DP0 pid=17822)         tensor_model_parallel_size,
(EngineCore_DP0 pid=17822)         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822)     ...<3 lines>...
(EngineCore_DP0 pid=17822)         backend,
(EngineCore_DP0 pid=17822)         ^^^^^^^^
(EngineCore_DP0 pid=17822)     )
(EngineCore_DP0 pid=17822)     ^
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/distributed/parallel_state.py", line 1369, in initialize_model_parallel
(EngineCore_DP0 pid=17822)     _TP = init_model_parallel_group(
(EngineCore_DP0 pid=17822)         group_ranks,
(EngineCore_DP0 pid=17822)     ...<3 lines>...
(EngineCore_DP0 pid=17822)         group_name="tp",
(EngineCore_DP0 pid=17822)     )
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/distributed/parallel_state.py", line 1089, in init_model_parallel_group
(EngineCore_DP0 pid=17822)     return GroupCoordinator(
(EngineCore_DP0 pid=17822)         group_ranks=group_ranks,
(EngineCore_DP0 pid=17822)     ...<4 lines>...
(EngineCore_DP0 pid=17822)         group_name=group_name,
(EngineCore_DP0 pid=17822)     )
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/distributed/parallel_state.py", line 362, in __init__
(EngineCore_DP0 pid=17822)     self.device_communicator = device_comm_cls(
(EngineCore_DP0 pid=17822)                                ~~~~~~~~~~~~~~~^
(EngineCore_DP0 pid=17822)         cpu_group=self.cpu_group,
(EngineCore_DP0 pid=17822)         ^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822)     ...<2 lines>...
(EngineCore_DP0 pid=17822)         unique_name=self.unique_name,
(EngineCore_DP0 pid=17822)         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822)     )
(EngineCore_DP0 pid=17822)     ^
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/distributed/device_communicators/cuda_communicator.py", line 58, in __init__
(EngineCore_DP0 pid=17822)     self.pynccl_comm = PyNcclCommunicator(
(EngineCore_DP0 pid=17822)                        ~~~~~~~~~~~~~~~~~~^
(EngineCore_DP0 pid=17822)         group=self.cpu_group,
(EngineCore_DP0 pid=17822)         ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822)         device=self.device,
(EngineCore_DP0 pid=17822)         ^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822)     )
(EngineCore_DP0 pid=17822)     ^
(EngineCore_DP0 pid=17822)   File "/opt/venv/lib64/python3.13/site-packages/vllm/distributed/device_communicators/pynccl.py", line 145, in __init__
(EngineCore_DP0 pid=17822)     data = torch.zeros(1, device=device)
(EngineCore_DP0 pid=17822) torch.AcceleratorError: HIP error: invalid kernel file
(EngineCore_DP0 pid=17822) Search for `hipErrorInvalidKernelFile' in https://rocm.docs.amd.com/projects/HIP/en/latest/index.html for more information.
(EngineCore_DP0 pid=17822) HIP kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
(EngineCore_DP0 pid=17822) For debugging consider passing AMD_SERIALIZE_KERNEL=3
(EngineCore_DP0 pid=17822) Device-side assertion tracking was not enabled by user.
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344] Error executing method 'init_device'. This might cause deadlock in distributed execution.
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344] Traceback (most recent call last):
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/worker/worker_base.py", line 334, in execute_method
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     return run_method(self, method, args, kwargs)
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/serial_utils.py", line 459, in run_method
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     return func(*args, **kwargs)
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]   File "/opt/venv/lib64/python3.13/site-packages/ray/util/tracing/tracing_helper.py", line 461, in _resume_span
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     return method(self, *_args, **_kwargs)
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/worker/worker_base.py", line 326, in init_device
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     self.worker.init_device()  # type: ignore
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     ~~~~~~~~~~~~~~~~~~~~~~~^^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/worker/gpu_worker.py", line 221, in init_device
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     init_worker_distributed_environment(
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]         self.vllm_config,
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]         ^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     ...<3 lines>...
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]         current_platform.dist_backend,
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     )
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     ^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/worker/gpu_worker.py", line 959, in init_worker_distributed_environment
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     ensure_model_parallel_initialized(
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]         parallel_config.tensor_parallel_size,
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     ...<2 lines>...
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]         parallel_config.decode_context_parallel_size,
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     )
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     ^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]   File "/opt/venv/lib64/python3.13/site-packages/vllm/distributed/parallel_state.py", line 1472, in ensure_model_parallel_initialized
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     initialize_model_parallel(
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     ~~~~~~~~~~~~~~~~~~~~~~~~~^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]         tensor_model_parallel_size,
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     ...<3 lines>...
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]         backend,
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]         ^^^^^^^^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     )
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     ^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]   File "/opt/venv/lib64/python3.13/site-packages/vllm/distributed/parallel_state.py", line 1369, in initialize_model_parallel
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     _TP = init_model_parallel_group(
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]         group_ranks,
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     ...<3 lines>...
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]         group_name="tp",
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     )
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]   File "/opt/venv/lib64/python3.13/site-packages/vllm/distributed/parallel_state.py", line 1089, in init_model_parallel_group
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     return GroupCoordinator(
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]         group_ranks=group_ranks,
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     ...<4 lines>...
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]         group_name=group_name,
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     )
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]   File "/opt/venv/lib64/python3.13/site-packages/vllm/distributed/parallel_state.py", line 362, in __init__
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     self.device_communicator = device_comm_cls(
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]                                ~~~~~~~~~~~~~~~^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]         cpu_group=self.cpu_group,
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]         ^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     ...<2 lines>...
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]         unique_name=self.unique_name,
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     )
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     ^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]   File "/opt/venv/lib64/python3.13/site-packages/vllm/distributed/device_communicators/cuda_communicator.py", line 58, in __init__
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     self.pynccl_comm = PyNcclCommunicator(
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]                        ~~~~~~~~~~~~~~~~~~^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]         group=self.cpu_group,
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]         ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]         device=self.device,
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]         ^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     )
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     ^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]   File "/opt/venv/lib64/python3.13/site-packages/vllm/distributed/device_communicators/pynccl.py", line 145, in __init__
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344]     data = torch.zeros(1, device=device)
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344] torch.AcceleratorError: HIP error: invalid kernel file
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344] Search for `hipErrorInvalidKernelFile' in https://rocm.docs.amd.com/projects/HIP/en/latest/index.html for more information.
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344] HIP kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344] For debugging consider passing AMD_SERIALIZE_KERNEL=3
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) ERROR 01-31 14:21:03 [worker_base.py:344] Device-side assertion tracking was not enabled by user.
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) WARNING 01-31 14:21:03 [worker_base.py:301] Missing `shared_worker_lock` argument from executor. This argument is needed for mm_processor_cache_type='shm'.
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=9409, ip=192.168.100.2) INFO 01-31 14:21:03 [parallel_state.py:1234] world_size=2 rank=1 local_rank=0 distributed_init_method=tcp://192.168.100.1:49483 backend=nccl
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344] Error executing method 'init_device'. This might cause deadlock in distributed execution.
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344] Traceback (most recent call last):
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/worker/worker_base.py", line 334, in execute_method
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]     return run_method(self, method, args, kwargs)
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/serial_utils.py", line 459, in run_method
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]     return func(*args, **kwargs)
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]   File "/opt/venv/lib64/python3.13/site-packages/ray/util/tracing/tracing_helper.py", line 461, in _resume_span
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]     return method(self, *_args, **_kwargs)
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/worker/gpu_worker.py", line 221, in init_device [repeated 2x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]     self.worker.init_device()  # type: ignore
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]     ~~~~~~~~~~~~~~~~~~~~~~~^^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]     init_worker_distributed_environment(
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]         self.vllm_config,
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]         ^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]     ...<2 lines>... [repeated 6x across cluster]
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]         current_platform.dist_backend,
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]     ) [repeated 7x across cluster]
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]     ^ [repeated 5x across cluster]
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]   File "/opt/venv/lib64/python3.13/site-packages/vllm/v1/worker/gpu_worker.py", line 959, in init_worker_distributed_environment
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]     ensure_model_parallel_initialized(
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]         parallel_config.tensor_parallel_size,
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]         parallel_config.decode_context_parallel_size,
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]   File "/opt/venv/lib64/python3.13/site-packages/vllm/distributed/parallel_state.py", line 1472, in ensure_model_parallel_initialized
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]     initialize_model_parallel(
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]     ~~~~~~~~~~~~~~~~~~~~~~~~~^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]         tensor_model_parallel_size,
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]         backend,
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]         ^^^^^^^^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]   File "/opt/venv/lib64/python3.13/site-packages/vllm/distributed/parallel_state.py", line 1369, in initialize_model_parallel
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]     _TP = init_model_parallel_group(
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]         group_ranks,
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]         group_name="tp",
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]   File "/opt/venv/lib64/python3.13/site-packages/vllm/distributed/parallel_state.py", line 1089, in init_model_parallel_group
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]     return GroupCoordinator(
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]         group_ranks=group_ranks,
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]         group_name=group_name,
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]   File "/opt/venv/lib64/python3.13/site-packages/vllm/distributed/device_communicators/pynccl.py", line 145, in __init__ [repeated 3x across cluster]
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]     self.device_communicator = device_comm_cls(
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]                                ~~~~~~~~~~~~~~~^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]         cpu_group=self.cpu_group,
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]         ^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]         unique_name=self.unique_name,
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]     self.pynccl_comm = PyNcclCommunicator(
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]                        ~~~~~~~~~~~~~~~~~~^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]         group=self.cpu_group,
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]         ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]         device=self.device,
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]         ^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344]     data = torch.zeros(1, device=device)
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344] torch.AcceleratorError: HIP error: invalid kernel file
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344] Search for `hipErrorInvalidKernelFile' in https://rocm.docs.amd.com/projects/HIP/en/latest/index.html for more information.
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344] HIP kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344] For debugging consider passing AMD_SERIALIZE_KERNEL=3
(EngineCore_DP0 pid=17822) (RayWorkerWrapper pid=17931) ERROR 01-31 14:21:03 [worker_base.py:344] Device-side assertion tracking was not enabled by user.

```

### 4.1 - Possible reasons

This invalid kernel file might be related to RCCL not supporting gfx1151. There was a PR that was never merged:

https://github.com/ROCm/rccl/pull/2075

Instead of being merged, this was closed when AMD moved ROCm to rocm-systems. Right now there is a feature request:

https://github.com/ROCm/rocm-systems/issues/2788

Again, I do not know if this is the issue or if the invalid kernel is a different issue altogether.