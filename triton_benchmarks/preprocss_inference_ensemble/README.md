# Benchmark: Triton Ensemble Model

## Goal

- Data flow on GPU
- Where should we run a preprocess? (CPU or GPU)
- How should we do image cropping?

## Setup

```bash
docker compose build
docker compose up -d
docker compose exec tritonserver bash
```

## Directory Layout

- model_repository
  - pattern0 (No Triton)
  - pattern1 (PythonBackend)
  - pattern2 (ensemble models) ... Preproc (Python) + Inference (Python) + Postproc (Python)
  - pattern3 (ensemble models) ... Preproc (Python) + Inference (ONNX) + Postproc (Python)
  - pattern4 (ensemble models) ... Preproc (Python) + Inference (TRT) + Postproc (Python)
  - detr_preproc (Python Backend)
  - detr_postproc (Python Backend)
  - detr_onnx (Onnx Runtime)
  - detr_trt (Tensor RT)
