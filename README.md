# AMD Strix Halo (gfx1151) — vLLM Toolbox/Container

An **Fedora 43** Docker/Podman container that is **Toolbx-compatible** (usable as a Fedora toolbox) for serving LLMs with **vLLM** on **AMD Ryzen AI Max “Strix Halo” (gfx1151)**. Built on the **TheRock nightly builds** for ROCm.

> **Note on Vision Models:** To prevent indefinite hangs during MIOpen exhaustive kernel search, vision encoder profiling is currently patched out. This means **vision features are currently unavailable** for models like Qwen3.5 MoE. Thanks to user `3spky5u-oss` for contributing this patch! For more details, see [Issue #30](https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes/issues/30).

---

## 🚀 High-Performance Clustering Support (New!)

**Update:** This toolbox now ships with a **custom build of ROCm/RCCL** that enables **native RDMA/RoCE v2 support for Strix Halo (gfx1151)**. This allows you to connect two nodes via a low-latency interconnect (e.g., Intel E810) and run vLLM with Tensor Parallelism (TP=2) effectively acting as a single 256GB Unified Memory GPU.

👉 **[Read the Full RDMA Cluster Setup Guide](rdma_cluster/setup_guide.md)** for hardware requirements and configuration instructions.

---

### 📦 Project Context

This repository is part of the **[Strix Halo AI Toolboxes](https://strix-halo-toolboxes.com)** project. Check out the website for an overview of all toolboxes, tutorials, and host configuration guides.

### 🙏 Acknowledgments

A big thank you to the [paudley/ai-notes](https://github.com/paudley/ai-notes) repository and its meticulously documented `strix-halo` build pipeline. I have directly integrated and adapted their research to stabilize vLLM and Flash-Attention for Strix Halo. Specifically, this toolbox leverages their Triton `AttrsDescriptor` compilation patches and the manual `aiter` compilation strategy that bridges CDNA kernels seamlessly over to RDNA scalar C++ routines—enabling full AITER custom op acceleration (MoE, Attention, RMSNorm) out-of-the-box.

### ❤️ Support

This is a hobby project maintained in my spare time. If you find these toolboxes and tutorials useful, you can **[buy me a coffee](https://buymeacoffee.com/dcapitella)** to support the work! ☕

---

## Table of Contents

* [Tested Models (Benchmarks)](#tested-models-benchmarks)
* [1) Toolbx vs Docker/Podman](#1-toolbx-vs-dockerpodman)
* [2) Quickstart — Fedora Toolbx](#2-quickstart--fedora-toolbx)
* [3) Quickstart — Ubuntu (Distrobox)](#3-quickstart--ubuntu-distrobox)
* [4) Testing the API](#4-testing-the-api)
* [5) Use a Web UI for Chatting](#5-use-a-web-ui-for-chatting)
* [6) Host Configuration](#6-host-configuration)
* [7) Distributed Clustering (RDMA/RoCE)](#7-distributed-clustering-rdmaroce)


## Tested Models (Benchmarks)

View full benchmarks at: [https://kyuz0.github.io/amd-strix-halo-vllm-toolboxes/](https://kyuz0.github.io/amd-strix-halo-vllm-toolboxes/)

**Table Key:** Cell values represent `Max Context Length (GPU Memory Utilization)`.

| Model | TP | 1 Req | 4 Reqs | 8 Reqs | 16 Reqs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`meta-llama/Meta-Llama-3.1-8B-Instruct`** | 1 | 128k (0.95) | 128k (0.95) | 128k (0.95) | 128k (0.95) |
| **`google/gemma-3-12b-it`** | 1 | 128k (0.95) | 128k (0.95) | 128k (0.95) | 128k (0.95) |
| **`openai/gpt-oss-20b`** | 1 | 128k (0.95) | 128k (0.95) | 128k (0.95) | 128k (0.95) |
| **`Qwen/Qwen3-14B-AWQ`** | 1 | 40k (0.95) | 40k (0.95) | 40k (0.95) | 40k (0.95) |
| **`btbtyler09/Qwen3-Coder-30B-A3B-Instruct-gptq-4bit`** | 1 | 256k (0.95) | 256k (0.95) | 256k (0.95) | 256k (0.95) |
| **`btbtyler09/Qwen3-Coder-30B-A3B-Instruct-gptq-8bit`** | 1 | 256k (0.95) | 256k (0.95) | 256k (0.95) | 256k (0.95) |
| **`dazipe/Qwen3-Next-80B-A3B-Instruct-GPTQ-Int4A16`** | 1 | 256k (0.95) | 256k (0.95) | 256k (0.95) | 256k (0.95) |
| **`openai/gpt-oss-120b`** | 1 | 128k (0.95) | 128k (0.95) | 128k (0.95) | 128k (0.95) |
| **`zai-org/GLM-4.7-Flash`** | 1 | 198k (0.95) | 198k (0.95) | 198k (0.95) | 198k (0.95) |


---

## 1) Toolbx vs Docker/Podman

The `kyuz0/vllm-therock-gfx1151:latest` image can be used both as: 

* **Fedora Toolbx (recommended for development):** Toolbx shares your **HOME** and user, so models/configs live on the host. Great for iterating quickly while keeping the host clean. 
* **Docker/Podman (recommended for deployment/perf):** Use for running vLLM as a service (host networking, IPC tuning, etc.). Always **mount a host directory** for model weights so they stay outside the container.

---

## 2) Quickstart — Fedora Toolbx

**Recommended:** Use the included `refresh_toolbox.sh` script. It pulls the latest image and creates the toolbox with the correct parameters:

```bash
./refresh_toolbox.sh
```

> **InfiniBand / RDMA Support:** The script automatically detects if a fast InfiniBand link is active (checks `/dev/infiniband`). If found, it correctly sets up the container to expose these devices, enabling high-performance clustering.

**Manual Creation:**

To manually create a toolbox that exposes the GPU and relaxes seccomp:

```bash
toolbox create vllm \
  --image docker.io/kyuz0/vllm-therock-gfx1151:latest \
  -- --device /dev/dri --device /dev/kfd \
  --group-add video --group-add render --security-opt seccomp=unconfined
```

Enter it:

```bash
toolbox enter vllm
```

**Model storage:** Models are downloaded to `~/.cache/huggingface` by default. This directory is shared with the host if you created the toolbox correctly, so downloads persist.

### Serving a Model (Easiest Way)

The toolbox includes a TUI wizard called **`start-vllm`** which includes pre-configured models and handles the launch flags for you. This is the easiest way to get started.

```bash
start-vllm
```

> **Cache note:** vLLM writes compiled kernels to `~/.cache/vllm/`.

---

## 3) Quickstart — Ubuntu (Distrobox)

Ubuntu’s toolbox package still breaks GPU access, so use Distrobox instead:

```bash
distrobox create -n vllm \
  --image docker.io/kyuz0/vllm-therock-gfx1151:latest \
  --additional-flags "--device /dev/kfd --device /dev/dri --group-add video --group-add render --security-opt seccomp=unconfined"

distrobox enter vllm
```

> **Verification:** Run `rocm-smi` to check GPU status.

### Serving a Model (Easiest Way)

The toolbox includes a TUI wizard called **`start-vllm`** which includes pre-configured models and handles the launch flags for you. This is the easiest way to get started.

```bash
start-vllm
```

---

## 4) Testing the API

Once the server is up, hit the OpenAI‑compatible endpoint:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"Hello! Test the performance."}]}'
```

You should receive a JSON response with a `choices[0].message.content` reply.

If you don't want to bother specifying the model name, you can run this which will query the currently deployed model:

```bash
MODEL=$(curl -s http://localhost:8000/v1/models | jq -r '.data[0].id') curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\":[{\"role\":\"user\",\"content\":\"Hello! Test the performance.\"}]
  }"
```

---

## 5) Use a Web UI for Chatting

If vLLM is on a remote server, expose port 8000 via SSH port forwarding:

```bash
ssh -L 0.0.0.0:8000:localhost:8000 <vllm-host>
```

Then, you can start HuggingFace ChatUI like this (on your host):

```bash
docker run -p 3000:3000 \
  --add-host=host.docker.internal:host-gateway \
  -e OPENAI_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=dummy \
  -v chat-ui-data:/data \
  ghcr.io/huggingface/chat-ui-db
```

## 6) Host Configuration

This should work on any Strix Halo. For a complete list of available hardware, see: [Strix Halo Hardware Database](https://strixhalo-homelab.d7.wtf/Hardware)

### 6.1 Test Configuration

| Component         | Specification                                               |
| :---------------- | :---------------------------------------------------------- |
| **Test Machine**  | Framework Desktop                                           |
| **CPU**           | Ryzen AI MAX+ 395 "Strix Halo"                              |
| **System Memory** | 128 GB RAM                                                  |
| **GPU Memory**    | 512 MB allocated in BIOS                                    |
| **Host OS**       | Fedora 43, Linux 6.18.5-200.fc43.x86_64            |

### 6.2 Kernel Parameters (tested on Fedora 42)

Add these boot parameters to enable unified memory while reserving a minimum of 4 GiB for the OS (max 124 GiB for iGPU):

iommu=pt amdgpu.gttsize=126976 ttm.pages_limit=32505856

| Parameter                   | Purpose                                                                                    |
|-----------------------------|--------------------------------------------------------------------------------------------|
| `iommu=pt`              | Sets IOMMU to "Pass-Through" mode. This helps performance, reducing overhead for both the RDMA NIC and the iGPU unified memory access.               |
| `amdgpu.gttsize=126976`     | Caps GPU unified memory to 124 GiB; 126976 MiB ÷ 1024 = 124 GiB                            |
| `ttm.pages_limit=32505856`  | Caps pinned memory to 124 GiB; 32505856 × 4 KiB = 126976 MiB = 124 GiB                     |

Source: https://www.reddit.com/r/LocalLLaMA/comments/1m9wcdc/comment/n5gf53d/?context=3&utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button


**Apply the changes:**

```
# Edit /etc/default/grub to add parameters to GRUB_CMDLINE_LINUX
sudo grub2-mkconfig -o /boot/grub2/grub.cfg
sudo reboot
```

## 7) Distributed Clustering (RDMA/RoCE)

This toolbox supports high-performance clustering of multiple Strix Halo nodes using Infiniband or RoCE v2 (e.g., Intel E810). This enables **Tensor Parallelism** across machines with extremely low latency (~5µs).

**Detailed Documentation:** [RDMA Cluster Setup Guide](rdma_cluster/setup_guide.md)

**Key Features:**
*   **Custom RCCL Patch:** Use of a custom-built `librccl.so` to support RDMA on `gfx1151`.
*   **Easy Setup:** `refresh_toolbox.sh` automatically detects and exposes RDMA devices.
*   **Cluster Management:** Included `start-vllm-cluster` TUI for managing Ray and vLLM.