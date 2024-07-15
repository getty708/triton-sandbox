# Overhead of Triton Ensemble Models

- Article: [Triton - Ensemble Models のオーバーヘッドの調査 (Zenn)](https://zenn.dev/getty708/articles/triton-ensemble-overhead)

## Goal

Triton ensemble models are a useful way to run multiple models. Connections between models are automatically handled by the triton.
However, we cannot optimize the connection between models because it is hidden by triton.
The goal of this project is to understand the overhead caused by Triton Ensemble Models compared to the monolithic pipeline.

## Prerequisites

- Docker with NVIDIA container toolkit.
- GPU Server

## How to run?

- Loanch docker container.

```bash
docker compose up -d
```

- In the first terminal, launch a triton server.

```bash
docker compose exec tritonserver bash

make start-triton
```

- In another terminal, send inference requests.

```bash
docker compose exec tritonserver bash

# Send request to the monolithic pipeline (# of models = 1)
make call-monolithic PIPELINE_STEPS=1
# Send request to the ensemble pipeline (# of models = 4)
make call-ensemble PIPELINE_STEPS=4
```

### (Optional) Launch triton Server with NVIDIA Nsight Systems (GPU Profiler)

- (Optional) Run triton sever with the profiler.

```bash
docker compose exec tritonserver bash

# Pipeline Architecture=`Monolithic`  (# of models = 1)
make start-triton-monolithic-with-nsys PIPELINE_STEPS=1
# Pipeline Architecture=`Ensemble` (# of models = 4)
make start-triton-monolithic-with-nsys PIPELINE_STEPS=4
```
