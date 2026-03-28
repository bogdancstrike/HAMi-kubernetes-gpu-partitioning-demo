# HAMi GPU Sharing on MicroK8s — Complete Tutorial

Split one physical GPU into multiple virtual slices shared across Kubernetes pods, using HAMi (Heterogeneous AI Computing Virtualization Middleware) on MicroK8s.

**Tested environment:**
- OS: Ubuntu 22.04 / 24.04
- GPU: NVIDIA GeForce RTX 3080 (10GB VRAM)
- Driver: 580.126.09 / CUDA 13.0
- MicroK8s: 1.32.x
- HAMi: v2.8.0 (hami-charts Helm)

**Files used in this tutorial:**
- `sanity_check.yaml` — GPU sanity check pod
- `gpu_worker_a.yaml` — Worker A deployment (light workload, 20% VRAM, 25% cores)
- `gpu_worker_b.yaml` — Worker B deployment (heavy workload, 30% VRAM, 40% cores)
- `hami_service_monitoring.yaml` — Prometheus ServiceMonitors for HAMi
- `grafana_dashboard.yaml` — Grafana dashboard ConfigMap

---

## How It Works

HAMi uses **CUDA API hijacking** via `LD_PRELOAD` — it intercepts `cudaMalloc` and kernel launch calls inside each container, enforcing per-pod VRAM and compute limits without any hardware-level partitioning (unlike MIG on A100/H100).

```
Physical GPU (e.g. RTX 3080 — 10GB VRAM)
├── Pod A  →  nvidia.com/gpu: 1 + nvidia.com/gpumem-percentage: 20  (~2048MB hard cap)
└── Pod B  →  nvidia.com/gpu: 1 + nvidia.com/gpumem-percentage: 30  (~3072MB hard cap)
             nvidia.com/gpucores: 25/40  (soft SM cap via kernel throttling)
```

Both pods run **truly in parallel** on different SMs when load allows. Time-slicing only occurs under SM contention.

---

## Prerequisites

- MicroK8s installed (`snap install microk8s --classic`)
- NVIDIA driver installed on the host (`nvidia-smi` works outside the cluster)
- `helm3` addon enabled in MicroK8s

---

## Step 1 — Enable the GPU Addon

MicroK8s ships a built-in GPU addon that installs the NVIDIA Container Toolkit and Device Plugin automatically:

```bash
microk8s enable gpu
```

Watch the GPU operator deploy:

```bash
microk8s kubectl get pods -n gpu-operator-resources -w
# Wait until all pods are Running
```

### Verify GPU is exposed to the cluster

```bash
microk8s kubectl get nodes -o json | \
  jq '.items[].status.capacity | with_entries(select(.key | startswith("nvidia")))'
```

Expected output:

```json
{
  "nvidia.com/gpu": "1"
}
```

### Sanity check — run nvidia-smi inside a pod

```bash
microk8s kubectl apply -f sanity_check.yaml
microk8s kubectl logs gpu-test
```

`sanity_check.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
spec:
  restartPolicy: Never
  containers:
  - name: cuda-test
    image: nvidia/cuda:12.1.0-base-ubuntu22.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: "1"
```

Expected output: your GPU name, driver version, CUDA version. Clean up:

```bash
microk8s kubectl delete pod gpu-test
```

---

## Step 2 — Install cert-manager

HAMi uses a mutating webhook that requires cert-manager for TLS certificate management.

```bash
microk8s kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.14.4/cert-manager.yaml
```

Wait for all three cert-manager pods to be ready:

```bash
microk8s kubectl wait \
  --for=condition=ready pod \
  -l app.kubernetes.io/instance=cert-manager \
  -n cert-manager \
  --timeout=180s
```

Or watch manually:

```bash
microk8s kubectl get pods -n cert-manager -w
# Wait for all three to show 1/1 Running:
# cert-manager-xxxxxxxxx          1/1   Running
# cert-manager-cainjector-xxxxx   1/1   Running
# cert-manager-webhook-xxxxx      1/1   Running
```

> **Do not proceed until all three are Running.** HAMi's webhook registration will fail silently if cert-manager is not ready.

---

## Step 3 — Install HAMi

### Add the Helm repository

```bash
microk8s helm3 repo add hami-charts https://project-hami.github.io/HAMi/
microk8s helm3 repo update

# Verify
microk8s helm3 search repo hami
# Should show: hami-charts/hami
```

### Detect your Kubernetes version

```bash
K8S_VERSION=$(microk8s kubectl version -o json | python3 -c "
import sys, json, re
v = json.load(sys.stdin)['serverVersion']['gitVersion'].lstrip('v')
print(re.split(r'[+\-]', v)[0])
")
echo "K8s version: $K8S_VERSION"
# Example output: 1.32.2
```

### Install HAMi

```bash
microk8s helm3 install hami hami-charts/hami \
  --namespace kube-system \
  --set scheduler.kubeScheduler.imageTag=v${K8S_VERSION} \
  --set devicePlugin.nvidiaDriverPath=/usr/local/nvidia \
  --set scheduler.defaultSchedulerPolicy.gpuMemory=true \
  --set scheduler.defaultSchedulerPolicy.gpuCores=true
```

### Label the GPU node

The HAMi device-plugin DaemonSet uses a node selector `gpu=on`. Without this label the DaemonSet stays at `DESIRED: 0` and the device-plugin never starts — meaning HAMi cannot inject its CUDA shim into pods and all VRAM limits are unenforced.

```bash
# Replace <your-node-name> with your actual node name (check: microk8s kubectl get nodes)
microk8s kubectl label node <your-node-name> gpu=on

# Verify the DaemonSet becomes active
microk8s kubectl get daemonset -n kube-system hami-device-plugin
# DESIRED should now be 1, not 0

# Watch the device-plugin pod start
microk8s kubectl get pods -n kube-system -l app.kubernetes.io/component=hami-device-plugin -w
```

### Verify HAMi is running

```bash
microk8s kubectl get pods -n kube-system | grep hami
# Expected:
# hami-device-plugin-xxxxx   2/2   Running   ← two containers: device-plugin + vgpu-monitor
# hami-scheduler-xxxxx       2/2   Running   ← two containers: kube-scheduler + vgpu-scheduler-extender
```

### Verify the node exposes vGPU resources

```bash
microk8s kubectl describe node | grep -A 15 "Allocatable"
```

Expected (new resources added by HAMi):

```
Allocatable:
  nvidia.com/gpu:       1
  nvidia.com/gpumem:    10240     ← total VRAM in MB
  nvidia.com/gpucores:  100       ← 100% SM budget
```

---

## Step 4 — Deploy Two Workloads on the Same GPU

### Critical resource specification rules

There are two important rules for HAMi v2.8.0 that differ from what you might expect:

**Rule 1 — `nvidia.com/gpu: "1"` is required.** HAMi uses this as the trigger to identify NVIDIA device requests. Setting only `gpucores` and `gpumem` without `nvidia.com/gpu` causes HAMi to log "No device requests found" and skip CUDA shim injection entirely.

**Rule 2 — Use `nvidia.com/gpumem-percentage` instead of `nvidia.com/gpumem`.** Kubernetes treats bare integer resource values as byte quantities and silently converts `2000` to `2k` (2 kilobytes). HAMi then sees 2KB instead of 2000MB and considers the request trivial. The `gpumem-percentage` resource is a plain integer (percentage of total GPU VRAM) that Kubernetes does not reinterpret.

```yaml
resources:
  limits:
    nvidia.com/gpu: "1"                    # required — HAMi entry point for NVIDIA devices
    nvidia.com/gpucores: "25"              # % of SM cores (plain integer, no conversion)
    nvidia.com/gpumem-percentage: "20"     # % of total VRAM — use this, not nvidia.com/gpumem
```

### Resource budget planning (RTX 3080 example)

```
RTX 3080 — 10240 MB total VRAM
├── Desktop (Xorg + gnome-shell):   ~930 MB   (if running a desktop environment)
├── Worker A: 20% = 2048 MB
├── Worker B: 30% = 3072 MB
├── CUDA context overhead (2 pods):  ~430 MB
                                    ─────────
                                    ~6480 MB used  (~4GB headroom)
```

### Worker A — light workload

`gpu_worker_a.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gpu-worker-a
spec:
  replicas: 1
  selector:
    matchLabels:
      app: gpu-worker
      instance: worker-a
  template:
    metadata:
      labels:
        app: gpu-worker
        instance: worker-a
    spec:
      schedulerName: hami-scheduler
      containers:
      - name: gpu-worker
        image: pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
        command: ["python3", "-u", "-c"]
        args:
        - |
          import torch, time, os
          pod = os.environ.get('POD_NAME', 'worker-a')
          device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
          print(f'[{pod}] device={device} gpu={torch.cuda.get_device_name(0)}', flush=True)
          elements = (1500 * 1024 * 1024) // 4
          blob = torch.zeros(elements, dtype=torch.float32, device=device)
          used = torch.cuda.memory_allocated() // 1024**2
          print(f'[{pod}] VRAM allocated: {used}MB', flush=True)
          a = torch.randn(1024, 1024, device=device, dtype=torch.float16)
          b = torch.randn(1024, 1024, device=device, dtype=torch.float16)
          i = 0
          while True:
              c = torch.matmul(a, b)
              torch.cuda.synchronize()
              i += 1
              if i % 100 == 0:
                  used = torch.cuda.memory_allocated() // 1024**2
                  print(f'[{pod}] iter={i} vram={used}MB', flush=True)
              time.sleep(0.1)
        env:
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: PYTHONUNBUFFERED
          value: "1"
        resources:
          limits:
            nvidia.com/gpu: "1"              # required — HAMi uses this as the trigger
            nvidia.com/gpucores: "25"
            nvidia.com/gpumem-percentage: "20"
```

### Worker B — heavier workload

`gpu_worker_b.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gpu-worker-b
spec:
  replicas: 1
  selector:
    matchLabels:
      app: gpu-worker
      instance: worker-b
  template:
    metadata:
      labels:
        app: gpu-worker
        instance: worker-b
    spec:
      schedulerName: hami-scheduler
      containers:
      - name: gpu-worker
        image: pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
        command: ["python3", "-u", "-c"]
        args:
        - |
          import torch, time, os
          pod = os.environ.get('POD_NAME', 'worker-b')
          device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
          print(f'[{pod}] device={device} gpu={torch.cuda.get_device_name(0)}', flush=True)
          elements = (2000 * 1024 * 1024) // 4
          blob = torch.zeros(elements, dtype=torch.float32, device=device)
          used = torch.cuda.memory_allocated() // 1024**2
          print(f'[{pod}] VRAM allocated: {used}MB', flush=True)
          a = torch.randn(2048, 2048, device=device, dtype=torch.float16)
          b = torch.randn(2048, 2048, device=device, dtype=torch.float16)
          i = 0
          while True:
              c = torch.matmul(a, b)
              torch.cuda.synchronize()
              i += 1
              if i % 100 == 0:
                  used = torch.cuda.memory_allocated() // 1024**2
                  print(f'[{pod}] iter={i} vram={used}MB', flush=True)
              time.sleep(0.05)
        env:
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: PYTHONUNBUFFERED
          value: "1"
        resources:
          limits:
            nvidia.com/gpu: "1"              # required trigger
            nvidia.com/gpucores: "40"
            nvidia.com/gpumem-percentage: "30"
```

### Apply both deployments

```bash
microk8s kubectl apply -f gpu_worker_a.yaml
microk8s kubectl apply -f gpu_worker_b.yaml

# Watch pods come up (pytorch image is ~5GB — first pull takes a few minutes)
microk8s kubectl get pods -l app=gpu-worker -w
```

---

## Step 5 — Verify GPU Sharing

### Confirm HAMi injected its CUDA shim

```bash
POD_A=$(microk8s kubectl get pod -l instance=worker-a -o jsonpath='{.items[0].metadata.name}')
microk8s kubectl exec $POD_A -- env | grep CUDA_DEVICE_MEMORY_SHARED_CACHE
# Expected: CUDA_DEVICE_MEMORY_SHARED_CACHE=/usr/local/vgpu/<uuid>.cache
# Empty = shim not injected — see troubleshooting
```

### Confirm bind-phase completed successfully

```bash
microk8s kubectl get pods -l app=gpu-worker -o yaml | grep "bind-phase"
# Expected: hami.io/bind-phase: success
# "allocating" = device-plugin didn't complete injection — delete pods and reschedule
```

### Tail logs from both pods simultaneously

```bash
microk8s kubectl logs -l app=gpu-worker --prefix=true -f
```

Expected output:

```
[pod/gpu-worker-a-.../gpu-worker] [worker-a] device=cuda gpu=NVIDIA GeForce RTX 3080
[pod/gpu-worker-b-.../gpu-worker] [worker-b] device=cuda gpu=NVIDIA GeForce RTX 3080
[pod/gpu-worker-b-.../gpu-worker] [worker-b] VRAM allocated: 2000MB
[pod/gpu-worker-a-.../gpu-worker] [worker-a] VRAM allocated: 1500MB
[pod/gpu-worker-b-.../gpu-worker] [worker-b] iter=100 vram=2032MB
[pod/gpu-worker-a-.../gpu-worker] [worker-a] iter=100 vram=1514MB
```

### Confirm on the host with nvidia-smi

```bash
watch -n 2 nvidia-smi
```

Expected — two separate python3 processes both under **GPU 0**:

```
+-------------------------------------------------------------------------------------+
| Processes:                                                                          |
|  GPU   GI   CI   PID     Type  Process name                             GPU Memory |
|======================================================================================|
|    0   N/A  N/A  116033  C     python3                                    1828MiB  | ← worker-a
|    0   N/A  N/A  116034  C     python3                                    2860MiB  | ← worker-b
+-------------------------------------------------------------------------------------+
```

> **Note on iteration rates:** Worker B iterates faster than Worker A. This is the `gpucores` cap working — Worker B has 40% SM budget vs Worker A's 25%, so it completes more matrix multiplications per unit time.

---

## Step 6 — Monitoring with Prometheus and Grafana

HAMi ships two built-in Prometheus metric endpoints. No extra exporters are needed.

### HAMi's two metric endpoints

| NodePort | Service | Content |
|---|---|---|
| `:31992` | `hami-device-plugin-monitor` → pod `:9394` | Real-time per-container VRAM usage and SM utilization (from `vgpu-monitor` sidecar) |
| `:31993` | `hami-scheduler` → pod `:9395` | Allocation view — what HAMi committed to each pod and GPU |

Since this is a single-node MicroK8s cluster, both NodePorts are accessible directly from the host:

```bash
# Verify both are live with pods running
curl -s http://localhost:31992/metrics | grep -v "^#"
# Expected: vGPU_device_memory_usage_in_bytes, Device_utilization_desc_of_container per pod

curl -s http://localhost:31993/metrics | grep -v "^#"
# Expected: GPUDeviceSharedNum=2, GPUDeviceCoreAllocated=65, vGPUCoreAllocated per pod
```

### Enable the observability addon

```bash
microk8s enable observability

# Wait for all pods — takes 2-3 minutes
microk8s kubectl get pods -n observability -w
# All pods must be Running before applying ServiceMonitors
```

### Create ServiceMonitors

The HAMi Helm chart already creates two Services in `kube-system`. Wire them to Prometheus:

`hami_service_monitoring.yaml`:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: hami-scheduler-metrics
  namespace: observability
  labels:
    release: kube-prom-stack
spec:
  namespaceSelector:
    matchNames:
    - kube-system
  selector:
    matchLabels:
      app.kubernetes.io/component: hami-scheduler
      app.kubernetes.io/instance: hami
  endpoints:
  - port: monitor          # Service port name "monitor" → pod :9395
    interval: 10s
    path: /metrics
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: hami-device-plugin-metrics
  namespace: observability
  labels:
    release: kube-prom-stack
spec:
  namespaceSelector:
    matchNames:
    - kube-system
  selector:
    matchLabels:
      app.kubernetes.io/component: hami-device-plugin
      app.kubernetes.io/instance: hami
  endpoints:
  - port: monitorport      # Service port name "monitorport" → pod :9394
    interval: 5s
    path: /metrics
```

```bash
microk8s kubectl apply -f hami_service_monitoring.yaml
```

### Deploy the Grafana dashboard

The ConfigMap is auto-imported by Grafana within ~30 seconds when labelled `grafana_dashboard: "1"`:

`grafana_dashboard.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: hami-native-dashboard
  namespace: observability
  labels:
    grafana_dashboard: "1"
data:
  hami-native-dashboard.json: |
    {
      "title": "HAMi GPU Split — Native Metrics",
      "uid": "hami-native-v3",
      "refresh": "5s",
      "time": {"from": "now-15m", "to": "now"},
      "panels": [
        {
          "id": 1,
          "title": "VRAM Usage per Pod vs HAMi Limit (real-time)",
          "description": "Actual bytes each pod is using right now vs its hard limit",
          "type": "timeseries",
          "gridPos": {"x":0,"y":0,"w":12,"h":8},
          "targets": [
            {"expr": "vGPU_device_memory_usage_in_bytes{zone='vGPU'}", "legendFormat": "used — {{podname}}"},
            {"expr": "vGPU_device_memory_limit_in_bytes{zone='vGPU'}", "legendFormat": "limit — {{podname}}"}
          ],
          "fieldConfig": {"defaults": {"unit": "bytes"}}
        },
        {
          "id": 2,
          "title": "VRAM Usage % of HAMi Limit per Pod",
          "type": "timeseries",
          "gridPos": {"x":12,"y":0,"w":12,"h":8},
          "targets": [{
            "expr": "vGPU_device_memory_usage_in_bytes{zone='vGPU'} / vGPU_device_memory_limit_in_bytes{zone='vGPU'} * 100",
            "legendFormat": "{{podname}}"
          }],
          "fieldConfig": {"defaults": {"unit": "percent", "min": 0, "max": 100}}
        },
        {
          "id": 3,
          "title": "GPU Core Utilization per Pod (real-time %)",
          "description": "Device_utilization_desc_of_container from vgpu-monitor sidecar",
          "type": "timeseries",
          "gridPos": {"x":0,"y":8,"w":12,"h":8},
          "targets": [{
            "expr": "Device_utilization_desc_of_container{zone='vGPU'}",
            "legendFormat": "{{podname}}"
          }],
          "fieldConfig": {"defaults": {"unit": "percent"}}
        },
        {
          "id": 4,
          "title": "Physical GPU — Host Utilization & Total VRAM Used",
          "description": "Full GPU view as seen by the host — combines both pods and desktop",
          "type": "timeseries",
          "gridPos": {"x":12,"y":8,"w":12,"h":8},
          "targets": [
            {"expr": "HostCoreUtilization{zone='vGPU'}", "legendFormat": "SM util % — GPU {{deviceidx}}"},
            {"expr": "HostGPUMemoryUsage{zone='vGPU'} / 1073741824", "legendFormat": "VRAM GB — GPU {{deviceidx}}"}
          ],
          "fieldConfig": {"defaults": {"unit": "short"}}
        },
        {
          "id": 5,
          "title": "HAMi Allocated Cores per Pod (scheduler view)",
          "description": "vGPUCoreAllocated — what the scheduler committed, stays flat while pod runs",
          "type": "timeseries",
          "gridPos": {"x":0,"y":16,"w":12,"h":8},
          "targets": [{
            "expr": "vGPUCoreAllocated{zone='vGPU'}",
            "legendFormat": "{{podname}}"
          }],
          "fieldConfig": {"defaults": {"unit": "percent"}}
        },
        {
          "id": 6,
          "title": "HAMi Allocated VRAM per Pod (scheduler view)",
          "description": "vGPUMemoryAllocated — committed bytes per pod from scheduler perspective",
          "type": "timeseries",
          "gridPos": {"x":12,"y":16,"w":12,"h":8},
          "targets": [{
            "expr": "vGPUMemoryAllocated{zone='vGPU'}",
            "legendFormat": "{{podname}}"
          }],
          "fieldConfig": {"defaults": {"unit": "bytes"}}
        },
        {
          "id": 7,
          "title": "Containers Sharing This GPU",
          "type": "stat",
          "gridPos": {"x":0,"y":24,"w":6,"h":5},
          "targets": [{"expr": "GPUDeviceSharedNum{zone='vGPU'}", "legendFormat": "containers"}],
          "fieldConfig": {"defaults": {"thresholds": {"steps": [
            {"color": "green", "value": 0},
            {"color": "yellow", "value": 3},
            {"color": "red", "value": 7}
          ]}}}
        },
        {
          "id": 8,
          "title": "Total SM Cores Allocated",
          "type": "gauge",
          "gridPos": {"x":6,"y":24,"w":6,"h":5},
          "targets": [{"expr": "GPUDeviceCoreAllocated{zone='vGPU'}"}],
          "fieldConfig": {"defaults": {"unit": "percent", "min": 0, "max": 100,
            "thresholds": {"steps": [
              {"color": "green", "value": 0},
              {"color": "yellow", "value": 70},
              {"color": "red", "value": 90}
            ]}
          }}
        },
        {
          "id": 9,
          "title": "Total VRAM Allocated % of GPU",
          "type": "gauge",
          "gridPos": {"x":12,"y":24,"w":6,"h":5},
          "targets": [{"expr": "GPUDeviceMemoryAllocated{zone='vGPU'} / GPUDeviceMemoryLimit{zone='vGPU'} * 100"}],
          "fieldConfig": {"defaults": {"unit": "percent", "min": 0, "max": 100,
            "thresholds": {"steps": [
              {"color": "green", "value": 0},
              {"color": "yellow", "value": 60},
              {"color": "red", "value": 85}
            ]}
          }}
        },
        {
          "id": 10,
          "title": "Namespace GPU Quota Usage",
          "description": "Total gpumem (MB) and gpucores (%) consumed across all pods in default namespace",
          "type": "stat",
          "gridPos": {"x":18,"y":24,"w":6,"h":5},
          "targets": [
            {"expr": "QuotaUsed{quotaName='nvidia.com/gpucores',zone='vGPU'}", "legendFormat": "cores used"},
            {"expr": "QuotaUsed{quotaName='nvidia.com/gpumem',zone='vGPU'}", "legendFormat": "mem MB used"}
          ]
        }
      ],
      "schemaVersion": 38,
      "version": 1
    }
```

```bash
microk8s kubectl apply -f grafana_dashboard.yaml
```

### Access Grafana and Prometheus

```bash
# Get Grafana admin password
microk8s kubectl get secret -n observability kube-prom-stack-grafana \
  -o jsonpath='{.data.admin-password}' | base64 -d && echo

# Port-forward Grafana
microk8s kubectl port-forward -n observability svc/kube-prom-stack-grafana 3000:80 &

# Port-forward Prometheus (for direct PromQL)
microk8s kubectl port-forward -n observability svc/kube-prom-stack-kube-prome-prometheus 9292:9090 &
```

Open Grafana at `http://localhost:3000` (admin / password above), navigate to **Dashboards → Browse → HAMi GPU Split — Native Metrics**.

### Verify Prometheus targets are UP

```bash
# Check in browser: http://localhost:9292/targets
# Look for:
#   observability/hami-scheduler-metrics     → UP
#   observability/hami-device-plugin-metrics → UP

# Or via API
curl -s http://localhost:9292/api/v1/targets | \
  python3 -m json.tool | grep -E '"job"|"health"' | grep -B1 "hami"
```

### Dashboard panel reference

| Panel | Metric | Source port | What it shows |
|---|---|---|---|
| VRAM Usage per Pod | `vGPU_device_memory_usage/limit_in_bytes` | `:31992` | Real-time bytes used vs limit per pod |
| VRAM Usage % | ratio of above | `:31992` | How close each pod is to its hard cap |
| Core Utilization per Pod | `Device_utilization_desc_of_container` | `:31992` | SM compute % per pod in real time |
| Host GPU view | `HostCoreUtilization`, `HostGPUMemoryUsage` | `:31992` | Full physical GPU as nvidia-smi sees it |
| Allocated Cores per Pod | `vGPUCoreAllocated` | `:31993` | Flat line — what scheduler committed |
| Allocated VRAM per Pod | `vGPUMemoryAllocated` | `:31993` | Flat line — scheduler reservation |
| Containers on GPU | `GPUDeviceSharedNum` | `:31993` | How many pods share this GPU |
| Core/VRAM gauges | `GPUDeviceCoreAllocated`, `GPUDeviceMemoryAllocated` | `:31993` | Total committed on the GPU |
| Quota usage | `QuotaUsed` | `:31993` | Namespace-level consumption |

### Useful PromQL queries

```promql
# Real-time VRAM used per pod
vGPU_device_memory_usage_in_bytes

# VRAM as % of HAMi limit per pod
vGPU_device_memory_usage_in_bytes / vGPU_device_memory_limit_in_bytes * 100

# SM core utilization per pod
Device_utilization_desc_of_container

# Physical GPU — full host view
HostCoreUtilization
HostGPUMemoryUsage

# Scheduler allocation view (flat while pods run)
vGPUCoreAllocated
vGPUMemoryAllocated

# How many containers share this GPU right now
GPUDeviceSharedNum

# Total cores and VRAM committed on the GPU
GPUDeviceCoreAllocated
GPUDeviceMemoryAllocated / GPUDeviceMemoryLimit * 100
```

---

## Understanding the Limits

| Resource | Type | Enforcement | Behaviour on violation |
|---|---|---|---|
| `nvidia.com/gpumem-percentage` | Hard | HAMi intercepts `cudaMalloc` | Process OOM-killed immediately |
| `nvidia.com/gpucores` | Soft | Kernel submission throttling | Throughput reduced, not killed |

### Why VRAM is hard but cores are soft

VRAM is **spatially partitioned** — each pod's allocations live at fixed addresses. HAMi tracks cumulative `cudaMalloc` calls per process and kills the process the moment it would exceed its budget.

SM cores are **temporally shared** — HAMi injects `cudaDeviceSynchronize()` + sleep cycles into the container's CUDA call stream to slow down kernel submissions when utilization exceeds the cap. This is best-effort, not a hardware guarantee. Expect ±5–10% deviation from the configured cap.

---

## VRAM Sizing Reference

### Adjust limits for your GPU

| GPU | Total VRAM | Usable (approx) | Notes |
|---|---|---|---|
| RTX 3060 | 12 GB | ~11 GB | Consumer card, bare metal only |
| RTX 3080 | 10 GB | ~9 GB | Desktop eats ~930MB if running GUI |
| RTX 3090 | 24 GB | ~23 GB | |
| L40S | 48 GB | ~47 GB | No MIG, HAMi is primary split option |
| A100 40GB | 40 GB | ~39 GB | MIG available as alternative |
| A100 80GB | 80 GB | ~79 GB | MIG available as alternative |

### CUDA context overhead

Each pod consumes ~200–500MB just for the CUDA runtime context, independent of your workload. Always budget for this:

```
usable_vram = total_vram - (num_pods × 300MB) - other_gpu_processes
```

---

## Troubleshooting

### HAMi device-plugin DaemonSet has DESIRED: 0

The DaemonSet nodeSelector requires `gpu=on`. No pods will be created until the node is labelled:

```bash
microk8s kubectl label node <your-node-name> gpu=on
microk8s kubectl get daemonset -n kube-system hami-device-plugin
# DESIRED should change from 0 to 1
```

### Pod bind-phase stuck at "allocating" / CUDA_DEVICE_MEMORY_SHARED_CACHE not set

The device-plugin was not running when the pods were first scheduled. Delete the pods so they reschedule with the device-plugin active:

```bash
microk8s kubectl delete pod -l app=gpu-worker
# Deployments recreate them automatically

# Verify shim is now injected
POD_A=$(microk8s kubectl get pod -l instance=worker-a -o jsonpath='{.items[0].metadata.name}')
microk8s kubectl exec $POD_A -- env | grep CUDA_DEVICE_MEMORY_SHARED_CACHE
# Must be non-empty for VRAM enforcement to work

microk8s kubectl get pods -l app=gpu-worker -o yaml | grep "bind-phase"
# Must show: hami.io/bind-phase: success
```

### HAMi logs "No device requests found" for every pod

The scheduler sees the pod but finds no NVIDIA device requests. Two common causes:

**Missing `nvidia.com/gpu: "1"`** — add it to the resource limits alongside `gpucores` and `gpumem-percentage`.

**Using `nvidia.com/gpumem` with a plain integer** — Kubernetes converts `2000` to `2k` (2KB). Use `nvidia.com/gpumem-percentage` instead.

To confirm what the scheduler is parsing, check its logs during pod creation:

```bash
microk8s kubectl logs -n kube-system \
  $(microk8s kubectl get pod -n kube-system -l app.kubernetes.io/component=hami-scheduler \
  -o jsonpath='{.items[0].metadata.name}') \
  -c vgpu-scheduler-extender --since=2m | grep "idx=\|device requests\|allocate success"
```

A successful scheduling looks like:

```
devices.go:510] "Resource requirements collected" requests=[{"NVIDIA":{"Nums":1,"Memreq":0,"MemPercentagereq":20,"Coresreq":25}}]
device.go:852]  "device allocate success" allocate device={"NVIDIA":[{"Usedmem":2048,"Usedcores":25}]}
```

### HAMi scheduler metrics show GPUDeviceSharedNum=0 after pods restart

The scheduler lost its allocation state after restarting. Delete and reschedule the pods:

```bash
microk8s kubectl delete pod -l app=gpu-worker
sleep 15
curl -s http://localhost:31993/metrics | grep "GPUDeviceSharedNum\|GPUDeviceCoreAllocated"
# GPUDeviceSharedNum should now be 2, GPUDeviceCoreAllocated should be 65
```

### Pod OOM-killed on startup

```
torch.cuda.OutOfMemoryError: Tried to allocate X GiB
```

Your PyTorch allocation target exceeds available VRAM. This happens when another pod or the desktop has already consumed most of the GPU memory. Reduce the Python allocation and leave at least 300MB headroom below the `gpumem-percentage` limit.

### HAMi pods not appearing after helm install

```bash
# Check if cert-manager was ready when HAMi was installed
microk8s kubectl get pods -n cert-manager
microk8s kubectl describe deployment hami-scheduler -n kube-system

# If cert-manager wasn't ready, reinstall HAMi after it stabilises
microk8s helm3 uninstall hami -n kube-system
microk8s kubectl wait --for=condition=ready pod \
  -l app.kubernetes.io/instance=cert-manager -n cert-manager --timeout=180s
# Then reinstall
```

### kubectl logs fails with TLS certificate error

```
x509: certificate is valid for 192.168.1.164, not 192.168.1.187
```

Your GPU node changed IP (DHCP). Assign a static IP or always use `microk8s kubectl`:

```bash
microk8s config > ~/.kube/config
```

---

## Quick Reference — All Commands

```bash
# 1. Enable GPU addon
microk8s enable gpu

# 2. Install cert-manager
microk8s kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.14.4/cert-manager.yaml
microk8s kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance=cert-manager -n cert-manager --timeout=180s

# 3. Install HAMi
microk8s helm3 repo add hami-charts https://project-hami.github.io/HAMi/
microk8s helm3 repo update
K8S_VERSION=$(microk8s kubectl version -o json | python3 -c "import sys,json,re; v=json.load(sys.stdin)['serverVersion']['gitVersion'].lstrip('v'); print(re.split(r'[+\-]',v)[0])")
microk8s helm3 install hami hami-charts/hami --namespace kube-system \
  --set scheduler.kubeScheduler.imageTag=v${K8S_VERSION} \
  --set devicePlugin.nvidiaDriverPath=/usr/local/nvidia \
  --set scheduler.defaultSchedulerPolicy.gpuMemory=true \
  --set scheduler.defaultSchedulerPolicy.gpuCores=true

# 4. Label GPU node (required — device-plugin DaemonSet won't start without this)
microk8s kubectl label node <your-node-name> gpu=on

# 5. Deploy workloads
microk8s kubectl apply -f gpu_worker_a.yaml
microk8s kubectl apply -f gpu_worker_b.yaml

# 6. Verify GPU sharing
microk8s kubectl get pods -l app=gpu-worker -o yaml | grep "bind-phase"   # must be "success"
POD_A=$(microk8s kubectl get pod -l instance=worker-a -o jsonpath='{.items[0].metadata.name}')
microk8s kubectl exec $POD_A -- env | grep CUDA_DEVICE_MEMORY_SHARED_CACHE  # must be set
microk8s kubectl logs -l app=gpu-worker --prefix=true -f
watch -n 2 nvidia-smi

# 7. Verify HAMi native metrics
curl -s http://localhost:31992/metrics | grep -v "^#"   # real-time per-container
curl -s http://localhost:31993/metrics | grep -v "^#"   # scheduler allocation view

# 8. Enable monitoring
microk8s enable observability
microk8s kubectl get pods -n observability -w           # wait for all Running
microk8s kubectl apply -f hami_service_monitoring.yaml
microk8s kubectl apply -f grafana_dashboard.yaml

# 9. Access dashboards
microk8s kubectl get secret -n observability kube-prom-stack-grafana \
  -o jsonpath='{.data.admin-password}' | base64 -d && echo
microk8s kubectl port-forward -n observability svc/kube-prom-stack-grafana 3000:80 &
microk8s kubectl port-forward -n observability svc/kube-prom-stack-kube-prome-prometheus 9292:9090 &
# Grafana:    http://localhost:3000  → Dashboards → HAMi GPU Split — Native Metrics
# Prometheus: http://localhost:9292/targets
```
