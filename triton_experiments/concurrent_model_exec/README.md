# Concurrent Model Execution

## How to run?

- (1) Setup ONNX files.

```bash
make build
```

- (2) Launch the Triton Inference Server. Set workspace size of Triton Inference Server from `SHM_SIZE`. Default value in the make file is 8GB.

```bash
make run
```

When the all models are successfully loaded, you can see following messages.

```txt
I0630 12:48:54.073905 127 server.cc:676]
+-----------------------------+---------+--------+
| Model                       | Version | Status |
+-----------------------------+---------+--------+
| ensemble_full_parallel      | 1       | READY  |
| ensemble_partially_parallel | 1       | READY  |
| ensemble_sequential         | 1       | READY  |
| simple_cnn_l1               | 1       | READY  |
| simple_cnn_l2               | 1       | READY  |
| simple_cnn_l3               | 1       | READY  |
+-----------------------------+---------+--------+

...

I0630 12:48:54.204058 127 grpc_server.cc:2519] Started GRPCInferenceService at 0.0.0.0:8001
I0630 12:48:54.204435 127 http_server.cc:4623] Started HTTPService at 0.0.0.0:8000
I0630 12:48:54.247399 127 http_server.cc:315] Started Metrics Service at 0.0.0.0:8002
```

- (3) Loaunch client and send 10 request one by one. Select the pipeline to send request from `MODEL_NAME`. Available choise is as follows.
  - `MODEL_NAME=ensemble_sequential`
  - `MODEL_NAME=ensemble_partially_parallel`
  - `MODEL_NAME=ensemble_full_parallel`

```bash
make call-ensemble-model MODEL_NAME=ensemble_sequential
```

## Blog Post

- [Triton - Concurrent Model Execution (モデル並行実行) の検証](https://zenn.dev/getty708/articles/triton-concurrent-model-execution)
