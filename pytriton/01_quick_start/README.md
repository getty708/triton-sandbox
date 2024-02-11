# PyTriton: Quick Start

- [Docs: Quick Start](https://triton-inference-server.github.io/pytriton/0.5.1/quick_start/)
- [GitHub: Linear model in PyTorch](https://github.com/triton-inference-server/pytriton/tree/v0.5.1/examples/linear_random_pytorch)

## Overview

- Run code snippet on "Quick Start"
- Add another toy model ("multiply-by-2", `y = 2 * x`) at the same time.

## Setup

```bash
docker compose up -d
docker exec -it triton-sandbox bash
```

```bash
pip install -r requirement.txt
```

## Demo

### Start Triton Server (`server.py`)

```bash
python server.py
```

### Health Check

```bash
# server status
curl -v localhost:8000/v2/health/live

# model status (linear)
# If it is ready, triton returns only a status code (200).
curl -v localhost:8000/v2/models/linear/ready

# model status ()
curl -v localhost:8000/v2/models/multiply-by-2/ready
```

### Inference (via HTTP)

結果を確認しやすいように，入力データを `[0.1, 0.2]` に変更 ([input1.json](./input1.json)).

Linear model:

```bash
curl -X POST \
  -H "Content-Type: application/json"  \
  -d @input1.json \
  localhost:8000/v2/models/linear/infer
```

Another toy model:

```bash
curl -X POST \
  -H "Content-Type: application/json"  \
  -d @input1.json \
  localhost:8000/v2/models/multiply-by-2/infer
```

### Inference (via Python Client)

```bash
python client.py
```

---

## Output Log

### `server.py`

```bash
$ python server.py
W0211 06:05:55.864786 2860 pinned_memory_manager.cc:237] Unable to allocate pinned system memory, pinned memory pool will not be available: CUDA driver version is insufficient for CUDA runtime version
I0211 06:05:55.864822 2860 cuda_memory_manager.cc:117] CUDA memory pool disabled
E0211 06:05:55.864870 2860 server.cc:243] CudaDriverHelper has not been initialized.
I0211 06:05:55.865021 2860 server.cc:606]
+------------------+------+
| Repository Agent | Path |
+------------------+------+
+------------------+------+

I0211 06:05:55.865034 2860 server.cc:633]
+---------+------+--------+
| Backend | Path | Config |
+---------+------+--------+
+---------+------+--------+

I0211 06:05:55.865039 2860 server.cc:676]
+-------+---------+--------+
| Model | Version | Status |
+-------+---------+--------+
+-------+---------+--------+

Error: Failed to initialize NVML
W0211 06:05:55.866076 2860 metrics.cc:738] DCGM unable to start: DCGM initialization error
I0211 06:05:55.866150 2860 metrics.cc:710] Collecting CPU metrics
I0211 06:05:55.866222 2860 tritonserver.cc:2483]
+----------------------------------+------------------------------------------+
| Option                           | Value                                    |
+----------------------------------+------------------------------------------+
| server_id                        | triton                                   |
| server_version                   | 2.41.0                                   |
| server_extensions                | classification sequence model_repository |
|                                  |  model_repository(unload_dependents) sch |
|                                  | edule_policy model_configuration system_ |
|                                  | shared_memory cuda_shared_memory binary_ |
|                                  | tensor_data parameters statistics trace  |
|                                  | logging                                  |
| model_repository_path[0]         | /root/.cache/pytriton/workspace_6zvypg2j |
| model_control_mode               | MODE_EXPLICIT                            |
| strict_model_config              | 0                                        |
| rate_limit                       | OFF                                      |
| pinned_memory_pool_byte_size     | 268435456                                |
| min_supported_compute_capability | 6.0                                      |
| strict_readiness                 | 1                                        |
| exit_timeout                     | 30                                       |
| cache_enabled                    | 0                                        |
+----------------------------------+------------------------------------------+

I0211 06:05:55.868928 2860 grpc_server.cc:2495] Started GRPCInferenceService at 0.0.0.0:8001
I0211 06:05:55.869108 2860 http_server.cc:4619] Started HTTPService at 0.0.0.0:8000
I0211 06:05:55.911391 2860 http_server.cc:282] Started Metrics Service at 0.0.0.0:8002
I0211 06:05:56.890427 2860 model_lifecycle.cc:461] loading: linear:1
I0211 06:05:58.206431 2860 python_be.cc:2363] TRITONBACKEND_ModelInstanceInitialize: linear_0_0 (CPU device 0)
I0211 06:05:58.450124 2860 model_lifecycle.cc:818] successfully loaded 'linear'
I0211 06:05:58.465282 2860 model_lifecycle.cc:461] loading: multiply-by-2:1
I0211 06:05:59.747045 2860 python_be.cc:2363] TRITONBACKEND_ModelInstanceInitialize: multiply-by-2_0_0 (CPU device 0)
I0211 06:05:59.973116 2860 model_lifecycle.cc:818] successfully loaded 'multiply-by-2'
```

```bash
$ curl -v localhost:8000/v2/health/live
*   Trying 127.0.0.1:8000...
* Connected to localhost (127.0.0.1) port 8000 (#0)
> GET /v2/health/live HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.81.0
> Accept: */*
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Content-Length: 0
< Content-Type: text/plain
<
* Connection #0 to host localhost left intact
```

### Inference (HTTP)

Request:

```bash
$ curl -X POST \
  -H "Content-Type: application/json"  \
  -d @input1.json \
  localhost:8000/v2/models/linear/infer
```

Output:

```json
{
  "id": "0",
  "model_name": "linear",
  "model_version": "1",
  "outputs": [
    {
      "name": "OUTPUT_1",
      "datatype": "FP32",
      "shape": [1, 3],
      "data": [0.551192045211792, 0.48511916399002077, -0.45382392406463625]
    }
  ]
}
```

Request (Toy Model):

```bash
$ curl -X POST \
  -H "Content-Type: application/json"  \
  -d @input1.json \
  localhost:8000/v2/models/multiply-by-2/infer
```

Output: (まるめ誤差と思われるものが発生している.)

```json
{
  "id": "0",
  "model_name": "multiply-by-2",
  "model_version": "1",
  "outputs": [
    {
      "name": "OUTPUT_1",
      "datatype": "FP32",
      "shape": [1, 2],
      "data": [0.20000000298023225, 0.4000000059604645]
    }
  ]
}
```

### Inference (`client.py`)

```bash
root@bb35055f69a2:/workspace/triton-sandbox# python client.py
[INPUT] shape: (8, 2)
[INPUT] data:
 [[ 0.  1.]
 [ 2.  3.]
 [ 4.  5.]
 [ 6.  7.]
 [ 8.  9.]
 [10. 11.]
 [12. 13.]
 [14. 15.]]
=== liner ===
result_dict: ['OUTPUT_1']
[OUTPUT_1] shape: (8, 3)
[OUTPUT_1] data:
 [[ 0.26479095  0.7798872  -0.15354097]
 [-0.66129714  0.8658416   0.04496327]
 [-1.5873852   0.95179605  0.24346748]
 [-2.5134735   1.0377505   0.4419717 ]
 [-3.4395614   1.1237049   0.640476  ]
 [-4.3656497   1.2096592   0.8389802 ]
 [-5.291738    1.2956139   1.0374844 ]
 [-6.217826    1.3815682   1.2359887 ]]
=== Toy Model (multiply-by-2) ===
result_dict: ['OUTPUT_1']
[OUTPUT_1] shape: (8, 2)
[OUTPUT_1] data:
 [[ 0.  2.]
 [ 4.  6.]
 [ 8. 10.]
 [12. 14.]
 [16. 18.]
 [20. 22.]
 [24. 26.]
 [28. 30.]]
```
