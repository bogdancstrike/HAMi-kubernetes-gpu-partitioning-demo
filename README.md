# HAMi GPU Sharing on MicroK8s

Split a single physical NVIDIA GPU across multiple Kubernetes pods using [HAMi](https://github.com/Project-HAMi/HAMi) (Heterogeneous AI Computing Virtualization Middleware). No MIG, no hardware partitioning — works on consumer GPUs.

## How It Works

HAMi injects a CUDA shim via `LD_PRELOAD` into each container. The shim intercepts `cudaMalloc` and kernel launch calls to enforce per-pod limits:

- **VRAM** — hard cap; pod is OOM-killed if it exceeds its allocation
- **GPU cores** — soft cap via kernel submission throttling (±5–10% deviation is normal)

```
Physical GPU (e.g. RTX 3080 — 10 GB VRAM)
├── gpu-worker-a  →  20% VRAM (~2 GB)  +  25% SM cores
└── gpu-worker-b  →  30% VRAM (~3 GB)  +  40% SM cores
```

Both pods run truly in parallel on different SMs. Time-slicing only occurs under SM contention.

## Prerequisites

- Ubuntu 22.04 / 24.04
- NVIDIA driver installed on host (`nvidia-smi` works)
- MicroK8s installed (`snap install microk8s --classic`)
- `helm3` addon enabled in MicroK8s

**Tested with:** RTX 3080 (10 GB), driver 580.x / CUDA 13.0, MicroK8s 1.32.x, HAMi v2.8.0

## Quick Start

See **[docs/hami-microk8s-tutorial.md](docs/hami-microk8s-tutorial.md)** for the full step-by-step guide. Summary:

```bash
# 1. Enable GPU addon
microk8s enable gpu

# 2. Install cert-manager (required by HAMi's mutating webhook)
microk8s kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.14.4/cert-manager.yaml
microk8s kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance=cert-manager -n cert-manager --timeout=180s

# 3. Install HAMi
microk8s helm3 repo add hami-charts https://project-hami.github.io/HAMi/
microk8s helm3 repo update
K8S_VERSION=$(microk8s kubectl version -o json | python3 -c "import sys,json,re; v=json.load(sys.stdin)['serverVersion']['gitVersion'].lstrip('v'); print(re.split(r'[+\-]',v)[0])")
microk8s helm3 install hami hami-charts/hami \
  --namespace kube-system \
  --set scheduler.kubeScheduler.imageTag=v${K8S_VERSION} \
  --set devicePlugin.nvidiaDriverPath=/usr/local/nvidia \
  --set scheduler.defaultSchedulerPolicy.gpuMemory=true \
  --set scheduler.defaultSchedulerPolicy.gpuCores=true

# 4. Label your GPU node (required — device-plugin DaemonSet uses nodeSelector gpu=on)
microk8s kubectl label node <your-node-name> gpu=on

# 5. Deploy workloads
microk8s kubectl apply -f manifests/workloads/sanity_check.yaml   # verify GPU is visible
microk8s kubectl apply -f manifests/workloads/gpu_worker_a.yaml
microk8s kubectl apply -f manifests/workloads/gpu_worker_b.yaml
```

## Files

```
manifests/
  workloads/
    sanity_check.yaml        # one-shot pod — runs nvidia-smi to verify GPU access
    gpu_worker_a.yaml        # light workload: 25% cores, 20% VRAM, 1024×1024 matmuls
    gpu_worker_b.yaml        # heavy workload: 40% cores, 30% VRAM, 2048×2048 matmuls
  monitoring/
    hami_service_monitoring.yaml   # Prometheus ServiceMonitors for HAMi
    grafana_dashboard.yaml         # Grafana ConfigMap — 10-panel GPU sharing dashboard
docs/
  hami-microk8s-tutorial.md  # full tutorial: install, verify, troubleshoot
```

## Resource Annotation Rules

```yaml
resources:
  limits:
    nvidia.com/gpu: "1"                  # required — HAMi uses this as the entry point
    nvidia.com/gpucores: "25"            # % of SM cores
    nvidia.com/gpumem-percentage: "20"   # % of total VRAM (do NOT use nvidia.com/gpumem)
```

> **Why `gpumem-percentage` and not `gpumem`?** Kubernetes treats bare integers as byte quantities and silently converts `2000` → `2k` (2 KB). HAMi then sees a 2 KB request and considers it trivial. The `-percentage` variant is not subject to this conversion.

All pods must also set `schedulerName: hami-scheduler`.

## Monitoring

HAMi exposes Prometheus metrics on two ports:

| Endpoint | Port | Content |
|----------|------|---------|
| vgpu-monitor (real-time per-pod) | `:31992` | `vGPU_device_memory_usage_in_bytes`, `Device_utilization_desc_of_container` |
| scheduler allocation view | `:31993` | `vGPUCoreAllocated`, `vGPUMemoryAllocated`, `GPUDeviceSharedNum` |

Apply `manifests/monitoring/hami_service_monitoring.yaml` and `manifests/monitoring/grafana_dashboard.yaml` to enable the Grafana dashboard.
