# Triton - ONNX Backend

## Goal

- [x] Deploy ONNX models on the triton inference server.
- [x] Build a pipeline of ONNX models using the triton ensemble models and check the data flow.
- [ ] Check options to optimize ONNX serving.

## Prerequisites & Environment

- Docker with NVIDIA container toolkit.
- GPU Server (GCP g2-standard-4)
  - NVIDIA L4 x 1
  - CPU: Intel Cascade Lake

## Models on Triton

- [`simple_cnn`](./models/simple_cnn.py): 3-layer CNN model.
- [`simple_transformer`](./models/simple_transformer.py): 2-layer transformer encoder.
  - embedding size: 2048
  - input sequence length: 32
- `simple_transformer_trt`: ORT-TRT optimization.
- `ensemble_cnn_4`: A pipeline that serially connects four Simple CNNs.
- `ensemble_transformer_4`: A pipeline that serially connects four Simple Transformers.

## How to run?

- Loanch docker container.

```bash
docker compose up -d
```

- In the first terminal, launch a triton server.

```bash
docker compose exec tritonserver bash

# Generate ONNX files
make init

# Start Triton Inference Server
make start-triton
```

- In another terminal, send inference requests.

```bash
docker compose exec tritonserver bash

# Send request to the Simple CNN (# of models = 1)
make call-monolithic-cnn
# Send request to the Simple Transformer (# of models = 1)
make call-monolithic-transformer
# Send request to the Simple Transformer (# of models = 1, with TensorRT Optimization)
make call-monolithic-transformer-trt
```
